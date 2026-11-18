#include "server-mcp-music.h"

#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

#include "base64.hpp"
#include "build-info.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// MCP over streamable HTTP, spec revisions this endpoint speaks (newest first)
static const char * k_protocol_versions[] = { "2025-06-18", "2025-03-26", "2024-11-05" };

static constexpr int    k_rpc_parse_error      = -32700;
static constexpr int    k_rpc_invalid_request  = -32600;
static constexpr int    k_rpc_method_not_found = -32601;
static constexpr int    k_rpc_invalid_params   = -32602;

// MiniMax-Music3 makes 25 audio frames per second (n_predict counts frames)
static constexpr double k_frames_per_second = 25.0;
static constexpr double k_default_seconds   = 30.0;
static constexpr double k_min_seconds       = 5.0;
static constexpr double k_max_seconds       = 360.0;
// songs whose WAV stays in memory; older ones keep only their file (or are forgotten when there is none)
static constexpr size_t k_songs_in_memory   = 32;
static constexpr size_t k_jobs_kept         = 64;
// larger songs are not put inline into a tool result, only their url
static constexpr size_t k_inline_audio_max  = 48u * 1024 * 1024;
static constexpr int    k_list_default      = 20;
static constexpr int    k_list_max          = 200;

static double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static std::string local_time(const char * fmt) {
    const std::time_t t = std::time(nullptr);
    std::tm tm {};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), fmt, &tm);
    return buf;
}

static double round2(double v) {
    return std::round(v * 100.0) / 100.0;
}

struct wav_info {
    int    sample_rate = 0;
    int    channels    = 0;
    int    bits        = 0;
    size_t data_bytes  = 0;

    double seconds() const {
        const double bytes_per_second = (double) sample_rate * channels * (bits / 8);
        return bytes_per_second > 0 ? data_bytes / bytes_per_second : 0.0;
    }
};

static uint32_t read_u32le(const std::string & b, size_t p) {
    return (uint32_t) (uint8_t) b[p] | ((uint32_t) (uint8_t) b[p + 1] << 8) | ((uint32_t) (uint8_t) b[p + 2] << 16) |
           ((uint32_t) (uint8_t) b[p + 3] << 24);
}

static uint16_t read_u16le(const std::string & b, size_t p) {
    return (uint16_t) ((uint8_t) b[p] | ((uint8_t) b[p + 1] << 8));
}

static bool parse_wav(const std::string & b, wav_info & w) {
    if (b.size() < 12 || b.compare(0, 4, "RIFF") != 0 || b.compare(8, 4, "WAVE") != 0) {
        return false;
    }
    size_t p = 12;
    while (p + 8 <= b.size()) {
        const std::string id = b.substr(p, 4);
        const size_t      sz = read_u32le(b, p + 4);
        if (id == "fmt " && sz >= 16 && p + 8 + 16 <= b.size()) {
            w.channels    = read_u16le(b, p + 10);
            w.sample_rate = (int) read_u32le(b, p + 12);
            w.bits        = read_u16le(b, p + 22);
        } else if (id == "data") {
            w.data_bytes = std::min(sz, b.size() - p - 8);
            break;
        }
        p += 8 + sz + (sz & 1);
    }
    return w.sample_rate > 0 && w.channels > 0 && w.bits > 0;
}

static std::string trim(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// cut at a character boundary of UTF-8 text
static std::string utf8_prefix(const std::string & s, size_t max_bytes) {
    if (s.size() <= max_bytes) {
        return s;
    }
    size_t n = max_bytes;
    while (n > 0 && ((uint8_t) s[n] & 0xC0) == 0x80) {
        n--;
    }
    return s.substr(0, n);
}

// a title from the first sung line, else from the start of the caption
static std::string default_title(const std::string & caption, const std::string & lyrics) {
    std::istringstream in(lyrics);
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (!line.empty() && line[0] != '[') {
            return utf8_prefix(line, 60);
        }
    }
    std::string t = trim(caption);
    const size_t nl = t.find('\n');
    if (nl != std::string::npos) {
        t = t.substr(0, nl);
    }
    return t.empty() ? "Untitled" : utf8_prefix(t, 60);
}

static std::string file_slug(const std::string & title) {
    std::string out;
    for (const char c : title) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_') {
            out += c;
        } else if ((c == ' ' || c == '.') && !out.empty() && out.back() != '-') {
            out += '-';
        }
        if (out.size() >= 40) {
            break;
        }
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out.empty() ? "song" : out;
}

static std::string header_value(const std::map<std::string, std::string> & headers, const std::string & name) {
    for (const auto & [k, v] : headers) {
        if (k.size() == name.size() && std::equal(k.begin(), k.end(), name.begin(), [](char a, char b) {
                return std::tolower((unsigned char) a) == std::tolower((unsigned char) b);
            })) {
            return v;
        }
    }
    return "";
}

static json rpc_result(const json & id, const json & result) {
    return { {"jsonrpc", "2.0"}, {"id", id}, {"result", result} };
}

static json rpc_error(const json & id, int code, const std::string & message) {
    return { {"jsonrpc", "2.0"}, {"id", id}, {"error", { {"code", code}, {"message", message} }} };
}

static json text_content(const std::string & text) {
    return { {"type", "text"}, {"text", text} };
}

static json tool_error(const std::string & message) {
    json content = json::array();
    content.push_back(text_content(message));
    return { {"content", content}, {"isError", true} };
}

static std::string sse_message(const json & msg) {
    return "event: message\ndata: " + msg.dump() + "\n\n";
}

// the message of an error that server_routes::run_tts returns as the JSON body of the /tts error
static std::string error_message(const std::string & err) {
    const json j = json::parse_no_throw(err);
    if (!j.is_discarded() && j.is_object()) {
        const json e = j.contains("error") ? j.at("error") : j;
        if (e.is_object() && e.contains("message") && e.at("message").is_string()) {
            return e.at("message").get<std::string>();
        }
    }
    return err;
}

struct music_song {
    std::string id;
    std::string title;
    std::string caption;
    std::string lyrics;
    std::string job_id;
    std::string file;
    std::string created;
    int64_t     seed           = 0;
    double      seconds        = 0;
    double      render_seconds = 0;
    int         sample_rate    = 0;
    int         channels       = 0;
    size_t      bytes          = 0;
    json        options;  // sampling options the request set
    std::shared_ptr<const std::string> wav; // null once dropped from memory, then read back from file
};

struct music_job {
    std::string id;
    std::string title;
    std::string caption;
    std::string lyrics;
    json        options;
    int64_t     seed      = 0;
    int32_t     n_predict = 0;
    double      seconds   = 0;  // requested length

    // guarded by impl::mtx
    std::string status = "queued"; // queued, running, done, failed, cancelled
    double      t_created  = 0;
    double      t_started  = 0;
    double      t_finished = 0;
    std::string song_id;
    std::string error;

    std::atomic<bool> cancel {false};
    std::shared_ptr<server_tts_progress> progress = std::make_shared<server_tts_progress>();

    bool finished() const {
        return status == "done" || status == "failed" || status == "cancelled";
    }
};

using music_job_ptr  = std::shared_ptr<music_job>;
using music_song_ptr = std::shared_ptr<music_song>;

struct server_mcp_music::impl {
    server_routes & routes;
    fs::path        dir;
    std::string     instance;  // 4 hex digits, so ids of one run do not repeat those of another

    std::mutex                  mtx;
    std::condition_variable     cv;  // any job changed state
    std::deque<music_job_ptr>   queue;
    std::vector<music_job_ptr>  jobs;   // oldest first
    std::vector<music_song_ptr> songs;  // oldest first
    uint64_t                    n_songs = 0;
    uint64_t                    n_jobs  = 0;
    std::atomic<bool>           stopping {false};
    std::thread                 worker;

    impl(const common_params & params, server_routes & routes) : routes(routes) {
        std::random_device rd;
        char buf[8];
        snprintf(buf, sizeof(buf), "%04x", (unsigned) (rd() & 0xffff));
        instance = buf;
        if (!params.mcp_music_dir.empty()) {
            dir = fs::u8path(params.mcp_music_dir);
            std::error_code ec;
            fs::create_directories(dir, ec);
            if (ec) {
                SRV_WRN("MCP music: cannot create %s (%s), songs stay in memory\n", params.mcp_music_dir.c_str(), ec.message().c_str());
                dir.clear();
            }
        }
        worker = std::thread([this]() { run(); });
    }

    ~impl() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stopping = true;
            for (auto & j : jobs) {
                j->cancel = true;
            }
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    //
    // worker
    //

    void run() {
        while (true) {
            music_job_ptr j;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&]() { return stopping.load() || !queue.empty(); });
                if (stopping) {
                    return;
                }
                j = queue.front();
                queue.pop_front();
                j->status    = "running";
                j->t_started = now_seconds();
            }
            cv.notify_all();

            json body = {
                {"input",           "[Tags]\n" + j->caption + "\n[Lyrics]\n" + j->lyrics},
                {"n_predict",       j->n_predict},
                {"seed",            j->seed},
                {"response_format", "wav"},
            };
            for (const auto & kv : j->options.items()) {
                body[kv.key()] = kv.value();
            }

            std::string audio;
            std::string err;
            const bool ok = routes.run_tts(body, j->progress, [this, j]() { return stopping.load() || j->cancel.load(); }, audio, err);

            music_song_ptr song;
            if (ok) {
                song = store_song(*j, std::move(audio), err);
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                j->t_finished = now_seconds();
                if (song) {
                    j->status  = "done";
                    j->song_id = song->id;
                } else if (j->cancel) {
                    j->status = "cancelled";
                } else {
                    j->status = "failed";
                    j->error  = error_message(err);
                }
            }
            if (song) {
                SRV_INF("MCP music: %s done, song %s (%.1f s of audio in %.1f s)\n", j->id.c_str(), song->id.c_str(), song->seconds,
                        song->render_seconds);
            } else {
                SRV_WRN("MCP music: %s %s%s%s\n", j->id.c_str(), j->status.c_str(), j->error.empty() ? "" : ": ", j->error.c_str());
            }
            cv.notify_all();
        }
    }

    music_song_ptr store_song(const music_job & j, std::string && audio, std::string & err) {
        wav_info w;
        if (!parse_wav(audio, w)) {
            err = "the model returned no WAV audio";
            return nullptr;
        }
        auto s = std::make_shared<music_song>();
        s->title          = j.title;
        s->caption        = j.caption;
        s->lyrics         = j.lyrics;
        s->job_id         = j.id;
        s->seed           = j.seed;
        s->options        = j.options;
        s->seconds        = w.seconds();
        s->sample_rate    = w.sample_rate;
        s->channels       = w.channels;
        s->bytes          = audio.size();
        s->created        = local_time("%Y-%m-%d %H:%M:%S");
        s->render_seconds = now_seconds() - j.t_started;

        std::lock_guard<std::mutex> lk(mtx);
        s->id = "song-" + instance + "-" + std::to_string(++n_songs);
        if (!dir.empty()) {
            const fs::path path = dir / fs::u8path(local_time("%Y%m%d-%H%M%S") + "_" + s->id + "_" + file_slug(s->title) + ".wav");
            std::ofstream f(path, std::ios::binary);
            if (f && f.write(audio.data(), (std::streamsize) audio.size())) {
                s->file = path.u8string();
            } else {
                SRV_WRN("MCP music: failed to write %s\n", path.u8string().c_str());
            }
        }
        s->wav = std::make_shared<const std::string>(std::move(audio));
        songs.push_back(s);

        // keep the newest k_songs_in_memory WAVs in memory, the files of older ones stay on disk
        size_t in_memory = 0;
        for (auto it = songs.rbegin(); it != songs.rend(); ++it) {
            if ((*it)->wav && ++in_memory > k_songs_in_memory) {
                (*it)->wav.reset();
            }
        }
        songs.erase(std::remove_if(songs.begin(), songs.end(), [](const music_song_ptr & x) { return !x->wav && x->file.empty(); }),
                    songs.end());
        return s;
    }

    // the WAV of a song, read back from its file when it left memory; caller holds no lock
    static std::shared_ptr<const std::string> load_wav(const music_song_ptr & s) {
        if (s->wav) {
            return s->wav;
        }
        if (s->file.empty()) {
            return nullptr;
        }
        std::ifstream f(fs::u8path(s->file), std::ios::binary);
        if (!f) {
            return nullptr;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        return std::make_shared<const std::string>(ss.str());
    }

    //
    // lookups (caller holds mtx)
    //

    // "latest", an id, an id with .wav, or a url that ends in one
    music_song_ptr find_song(std::string ref) const {
        ref = trim(ref);
        if (ref.empty() || ref == "latest" || ref == "last" || ref == "newest") {
            return songs.empty() ? nullptr : songs.back();
        }
        const size_t slash = ref.find_last_of('/');
        if (slash != std::string::npos) {
            ref = ref.substr(slash + 1);
        }
        const size_t q = ref.find_first_of("?#");
        if (q != std::string::npos) {
            ref = ref.substr(0, q);
        }
        if (ref.size() > 4 && ref.compare(ref.size() - 4, 4, ".wav") == 0) {
            ref = ref.substr(0, ref.size() - 4);
        }
        for (const auto & s : songs) {
            if (s->id == ref) {
                return s;
            }
        }
        return nullptr;
    }

    music_job_ptr find_job(const std::string & id) const {
        for (const auto & j : jobs) {
            if (j->id == id) {
                return j;
            }
        }
        return nullptr;
    }

    size_t queue_position(const music_job & j) const {
        for (size_t i = 0; i < queue.size(); ++i) {
            if (queue[i].get() == &j) {
                return i + 1;
            }
        }
        return 0;
    }

    //
    // reports
    //

    static std::string song_url(const music_song & s, const std::string & base_url) {
        return base_url + "/mcp/songs/" + s.id + ".wav";
    }

    static json describe(const music_song & s, const std::string & base_url, bool with_text) {
        json out = {
            {"id",             s.id},
            {"title",          s.title},
            {"seconds",        round2(s.seconds)},
            {"url",            song_url(s, base_url)},
            {"seed",           s.seed},
            {"sample_rate",    s.sample_rate},
            {"channels",       s.channels},
            {"created",        s.created},
            {"render_seconds", round2(s.render_seconds)},
            {"job_id",         s.job_id},
        };
        if (!s.file.empty()) {
            out["file"] = s.file;
        }
        if (!s.options.empty()) {
            out["options"] = s.options;
        }
        if (with_text) {
            out["caption"] = s.caption;
            out["lyrics"]  = s.lyrics;
        } else {
            out["caption"] = utf8_prefix(s.caption, 120);
        }
        return out;
    }

    struct progress_report {
        std::string text;
        double      percent = 0;
        json        state;
    };

    // caller holds mtx
    progress_report report(const music_job & j) const {
        const double now     = now_seconds();
        const auto & p       = *j.progress;
        const int    state   = p.state.load();
        const int    decoded = p.n_decoded.load();
        const int    total   = p.n_total.load();

        progress_report r;
        std::string phase = j.status;
        if (j.status == "queued") {
            const size_t pos = queue_position(j);
            r.text = "Waiting in the queue (position " + std::to_string(pos) + ")";
            phase  = "queued";
        } else if (j.status == "running") {
            if (state == server_tts_progress::GENERATING && total > 0) {
                const double done_s  = decoded / k_frames_per_second;
                const double total_s = total / k_frames_per_second;
                r.percent = 2.0 + 93.0 * std::min(1.0, (double) decoded / total);
                char buf[160];
                snprintf(buf, sizeof(buf), "Composing \"%s\": %.1f of %.0f s", j.title.c_str(), done_s, total_s);
                r.text = buf;
                phase  = "generating";
            } else if (state == server_tts_progress::FINISHING) {
                r.percent = 96.0;
                r.text    = "Finishing \"" + j.title + "\": last window and vocoder";
                phase     = "finishing";
            } else {
                r.percent = 1.0;
                r.text    = "Reading the caption and lyrics of \"" + j.title + "\"";
                phase     = "prompt";
            }
        } else if (j.status == "done") {
            r.percent = 100.0;
            r.text    = "Finished \"" + j.title + "\" (" + j.song_id + ")";
        } else {
            r.text = j.status == "cancelled" ? "Cancelled" : "Failed: " + j.error;
        }

        const double elapsed = j.t_started > 0 ? (j.finished() ? j.t_finished : now) - j.t_started : 0.0;
        json eta = nullptr;
        if (j.status == "running" && r.percent >= 5.0) {
            eta = std::round(elapsed * (100.0 - r.percent) / r.percent);
            r.text += ", about " + std::to_string((int) eta.get<double>()) + " s left";
        }
        r.state = {
            {"job_id",          j.id},
            {"status",          j.status},
            {"phase",           phase},
            {"percent",         round2(r.percent)},
            {"eta_seconds",     eta},
            {"elapsed_seconds", round2(elapsed)},
            {"title",           j.title},
            {"seconds",         round2(j.seconds)},
        };
        if (j.status == "running" && total > 0) {
            r.state["frames"] = { {"done", decoded}, {"total", total} };
        }
        if (j.status == "queued") {
            r.state["queue_position"] = (int64_t) queue_position(j);
        }
        if (!j.song_id.empty()) {
            r.state["song_id"] = j.song_id;
        }
        if (!j.error.empty()) {
            r.state["error"] = j.error;
        }
        return r;
    }

    // tool result with the song description and, if asked and small enough, its WAV as MCP audio content
    json song_result(const music_song_ptr & s, const std::string & base_url, bool with_audio, const std::string & note) {
        json desc = describe(*s, base_url, true);
        if (!note.empty()) {
            desc["message"] = note;
        }
        json content = json::array();
        std::shared_ptr<const std::string> wav;
        if (with_audio) {
            wav = load_wav(s);
            if (!wav) {
                desc["audio"] = "not available: the song is no longer in memory and has no file";
            } else if (wav->size() > k_inline_audio_max) {
                desc["audio"] = "too large to return inline (" + std::to_string(wav->size() >> 20) + " MiB), use the url";
                wav.reset();
            }
        }
        content.push_back(text_content(desc.dump(2)));
        if (wav) {
            content.push_back({ {"type", "audio"}, {"data", base64::encode(wav->data(), wav->size())}, {"mimeType", "audio/wav"} });
        }
        return { {"content", content} };
    }

    // caller holds mtx
    json job_result(const music_job & j, const std::string & base_url, bool with_audio) {
        if (j.status == "done") {
            const music_song_ptr s = find_song(j.song_id);
            if (s) {
                const std::string note = "Done. Give the user the url (" + song_url(*s, base_url) + ")" +
                    (with_audio ? "; the chat can also play the attached audio." : ".");
                return song_result(s, base_url, with_audio, note);
            }
            return tool_error("the song of " + j.id + " was deleted");
        }
        if (j.status == "cancelled") {
            return tool_error(j.id + " was cancelled before the song was finished");
        }
        return tool_error("generating the song failed: " + j.error);
    }

    //
    // MCP
    //

    json initialize_result(const json & params, const std::string & base_url) const {
        std::string version = k_protocol_versions[0];
        if (params.is_object() && params.contains("protocolVersion") && params.at("protocolVersion").is_string()) {
            const std::string requested = params.at("protocolVersion").get<std::string>();
            for (const char * v : k_protocol_versions) {
                if (requested == v) {
                    version = requested;
                }
            }
        }
        const json info = routes.tts_info();
        const std::string model = json_value(info, "model", std::string("a music model"));
        std::string instructions =
            "This server composes songs locally with a music model (" + model + "). Call generate_song with a caption (genre, mood, tempo "
            "or BPM, key, vocals, instruments and arrangement) and lyrics marked with section tags such as [intro], [verse], [chorus], "
            "[bridge] and [outro]. It returns the song as WAV audio that the chat can play, and its id and url: give the user the url. "
            "Songs render one at a time on the GPU and take about as long as the song plays, so a call may also wait in a queue; while "
            "it runs it reports progress and the time left. If a call times out, the job keeps running: job_status reports it, "
            "get_latest_song or get_song fetch the finished song, and cancel_job stops it. list_songs shows the songs made since the "
            "server started, get_song and get_latest_song return one again, delete_song removes songs.";
        if (!dir.empty()) {
            instructions += " Songs are saved as WAV files in " + dir.u8string() + "; delete_song also deletes the file.";
        } else {
            instructions += " Songs stay in memory only, until the server stops or delete_song removes them.";
        }
        return {
            {"protocolVersion", version},
            {"capabilities",    { {"tools", { {"listChanged", false} }} }},
            {"serverInfo",      { {"name", "llama-server-music"}, {"title", "llama.cpp music server"}, {"version", std::string(llama_build_info())} }},
            {"instructions",    instructions},
        };
    }

    static json annotations(bool read_only, bool destructive) {
        return { {"readOnlyHint", read_only}, {"destructiveHint", destructive}, {"idempotentHint", read_only}, {"openWorldHint", false} };
    }

    static json return_audio_schema() {
        return { {"type", "boolean"}, {"default", true},
                 {"description", "true returns the WAV so the chat can play it; false returns only the description and url."} };
    }

    json tools_list() const {
        json tools = json::array();

        json gen_props = {
            {"caption", { {"type", "string"},
                          {"description", "The style of the song in plain words: genre, mood, tempo or BPM, key, vocals (e.g. warm male "
                                          "lead, female duet) and the instruments and arrangement. Example: \"Genre: indie folk. BPM: 92. "
                                          "Key: G major. Vocals: warm male lead. Arrangement: acoustic guitar, light percussion and soft "
                                          "strings.\""} }},
            {"lyrics",  { {"type", "string"},
                          {"description", "The lyrics, one sung line per line, with section tags on their own lines: [intro], [verse], "
                                          "[pre-chorus], [chorus], [bridge], [outro]. Write as much as fits the length; the song ends "
                                          "when the model finishes or at the length limit."} }},
            {"seconds", { {"type", "number"}, {"minimum", k_min_seconds}, {"maximum", k_max_seconds}, {"default", k_default_seconds},
                          {"description", "Maximum length of the song in seconds; the model may end it earlier. Rendering takes about "
                                          "as long as the song."} }},
            {"title",   { {"type", "string"}, {"description", "A title for the song (used in listings and the file name)."} }},
            {"seed",    { {"type", "integer"}, {"minimum", 0},
                          {"description", "Random seed; the same seed, caption and lyrics give the same song. Omit for a new one."} }},
            {"temperature", { {"type", "number"}, {"minimum", 0}, {"maximum", 2},
                              {"description", "Sampling temperature of the audio codes; omit for the model default. 0 is greedy and "
                                              "can fall silent."} }},
            {"cfg_scale",   { {"type", "number"}, {"minimum", 0}, {"maximum", 20},
                              {"description", "How closely the music follows the caption (classifier-free guidance); omit for the "
                                              "model default."} }},
            {"return_audio", return_audio_schema()},
        };
        tools.push_back({
            {"name",        "generate_song"},
            {"title",       "Generate song"},
            {"description", "Compose and render a song with vocals from a style caption and lyrics. Returns the song as WAV audio the "
                            "chat can play, with its id, url, length and seed. Takes about as long as the song plays."},
            {"inputSchema", { {"type", "object"}, {"properties", gen_props}, {"required", json::array({"caption", "lyrics"})} }},
            {"annotations", annotations(false, false)},
        });
        tools.push_back({
            {"name",        "get_song"},
            {"title",       "Get song"},
            {"description", "Return a song made earlier, with its caption, lyrics, seed and url, and its audio."},
            {"inputSchema", { {"type", "object"},
                              {"properties", {
                                  {"song", { {"type", "string"},
                                             {"description", "Song id (e.g. \"song-1a2b-3\"), its url, or \"latest\" for the newest."} }},
                                  {"return_audio", return_audio_schema()},
                              }},
                              {"required", json::array({"song"})} }},
            {"annotations", annotations(true, false)},
        });
        tools.push_back({
            {"name",        "get_latest_song"},
            {"title",       "Get latest song"},
            {"description", "Return the newest finished song with its caption, lyrics, seed, url and audio. Use it to fetch a song whose "
                            "generate_song call timed out."},
            {"inputSchema", { {"type", "object"}, {"properties", { {"return_audio", return_audio_schema()} }} }},
            {"annotations", annotations(true, false)},
        });
        tools.push_back({
            {"name",        "list_songs"},
            {"title",       "List songs"},
            {"description", "List the songs this server made since it started, newest first, with id, title, length, caption, seed, "
                            "url and file."},
            {"inputSchema", { {"type", "object"},
                              {"properties", {
                                  {"limit", { {"type", "integer"}, {"minimum", 1}, {"maximum", k_list_max}, {"default", k_list_default},
                                              {"description", "Maximum number of songs to list."} }},
                              }} }},
            {"annotations", annotations(true, false)},
        });
        tools.push_back({
            {"name",        "delete_song"},
            {"title",       "Delete songs"},
            {"description", std::string("Delete songs: their ids and urls stop working") +
                            (dir.empty() ? "." : " and their WAV files are deleted from disk.") + " Only delete when the user asks."},
            {"inputSchema", { {"type", "object"},
                              {"properties", {
                                  {"songs", { {"type", "array"}, {"items", { {"type", "string"} }},
                                              {"description", "Ids or urls of the songs to delete; \"latest\" is the newest song."} }},
                                  {"all",   { {"type", "boolean"}, {"default", false},
                                              {"description", "true deletes every song this server holds."} }},
                              }} }},
            {"annotations", annotations(false, true)},
        });
        tools.push_back({
            {"name",        "job_status"},
            {"title",       "Job status"},
            {"description", "Report song jobs: queued (with queue position), running (with progress and time left), or finished (with "
                            "the song id). Use it when a generate_song call timed out: the job keeps running and its song can be "
                            "fetched with get_song or get_latest_song. Without job_id it lists the recent jobs, newest first."},
            {"inputSchema", { {"type", "object"},
                              {"properties", { {"job_id", { {"type", "string"},
                                                            {"description", "A job id (job-...). Omit to list recent jobs."} }} }} }},
            {"annotations", annotations(true, false)},
        });
        tools.push_back({
            {"name",        "cancel_job"},
            {"title",       "Cancel job"},
            {"description", "Stop a song job: a queued one is dropped, a running one stops after its current frame. Without job_id it "
                            "cancels the running job, or else the newest queued one. Only cancel when the user asks."},
            {"inputSchema", { {"type", "object"},
                              {"properties", { {"job_id", { {"type", "string"},
                                                            {"description", "The job to cancel (job-...). Omit for the running job."} }} }} }},
            {"annotations", annotations(false, true)},
        });
        tools.push_back({
            {"name",        "get_server_info"},
            {"title",       "Server info"},
            {"description", "Report the music model, the sample rate, the length limits, the queue and where songs are saved."},
            {"inputSchema", { {"type", "object"}, {"properties", json::object()} }},
            {"annotations", annotations(true, false)},
        });
        return { {"tools", tools} };
    }

    // checks the arguments of generate_song and queues the job; error is set when they are invalid
    music_job_ptr queue_song(const json & args, std::string & error) {
        auto get_str = [&](const char * key) {
            return args.contains(key) && args.at(key).is_string() ? args.at(key).get<std::string>() : std::string();
        };
        auto j = std::make_shared<music_job>();
        j->caption = trim(get_str("caption"));
        j->lyrics  = trim(get_str("lyrics"));
        if (j->caption.empty()) {
            error = "caption is required: describe the genre, mood, tempo, vocals and instruments";
            return nullptr;
        }
        if (!args.contains("lyrics") || !args.at("lyrics").is_string()) {
            error = "lyrics is required (a string with section tags such as [verse] and [chorus])";
            return nullptr;
        }
        double seconds = k_default_seconds;
        if (args.contains("seconds") && args.at("seconds").is_number()) {
            seconds = args.at("seconds").get<double>();
        }
        seconds = std::max(k_min_seconds, std::min(k_max_seconds, seconds));
        // the frames and the prompt share the backbone's context
        const json info = routes.tts_info();
        const int64_t n_ctx = json_value(info, "n_ctx", (int64_t) 0);
        if (n_ctx > 0) {
            seconds = std::min(seconds, std::max(k_min_seconds, (n_ctx - 1024) / k_frames_per_second));
        }
        j->seconds   = seconds;
        j->n_predict = (int32_t) std::lround(seconds * k_frames_per_second);

        if (args.contains("seed") && args.at("seed").is_number()) {
            j->seed = std::max<int64_t>(0, args.at("seed").get<int64_t>());
        } else {
            std::random_device rd;
            j->seed = (int64_t) (rd() & 0x7fffffff);
        }
        j->options = json::object();
        if (args.contains("temperature") && args.at("temperature").is_number()) {
            j->options["temperature"] = std::max(0.0, std::min(2.0, args.at("temperature").get<double>()));
        }
        if (args.contains("cfg_scale") && args.at("cfg_scale").is_number()) {
            j->options["cfg_scale"] = std::max(0.0, std::min(20.0, args.at("cfg_scale").get<double>()));
        }
        const std::string title = trim(get_str("title"));
        j->title = title.empty() ? default_title(j->caption, j->lyrics) : utf8_prefix(title, 120);

        {
            std::lock_guard<std::mutex> lk(mtx);
            j->id        = "job-" + instance + "-" + std::to_string(++n_jobs);
            j->t_created = now_seconds();
            jobs.push_back(j);
            // forget the oldest finished jobs
            while (jobs.size() > k_jobs_kept) {
                auto it = std::find_if(jobs.begin(), jobs.end(), [](const music_job_ptr & x) { return x->finished(); });
                if (it == jobs.end()) {
                    break;
                }
                jobs.erase(it);
            }
            queue.push_back(j);
        }
        cv.notify_all();
        SRV_INF("MCP music: queued %s \"%s\", %.0f s, seed %lld\n", j->id.c_str(), j->title.c_str(), seconds, (long long) j->seed);
        return j;
    }

    json call_tool(const std::string & name, const json & args, const std::string & base_url) {
        const bool with_audio = json_value(args, "return_audio", true);

        if (name == "get_song" || name == "get_latest_song") {
            const std::string ref = name == "get_latest_song" ? std::string("latest") : json_value(args, "song", std::string());
            music_song_ptr s;
            {
                std::lock_guard<std::mutex> lk(mtx);
                s = find_song(ref);
            }
            if (!s) {
                return tool_error(ref == "latest" || ref.empty() ? "no song has been made yet" : "no song \"" + ref + "\" (list_songs shows the ids)");
            }
            return song_result(s, base_url, with_audio, "");
        }
        if (name == "list_songs") {
            const int limit = std::max(1, std::min(k_list_max, json_value(args, "limit", k_list_default)));
            json list = json::array();
            size_t total = 0;
            {
                std::lock_guard<std::mutex> lk(mtx);
                total = songs.size();
                for (auto it = songs.rbegin(); it != songs.rend() && (int) list.size() < limit; ++it) {
                    list.push_back(describe(**it, base_url, false));
                }
            }
            const json out = { {"songs", list}, {"count", (int64_t) list.size()}, {"total", (int64_t) total} };
            json content = json::array();
            content.push_back(text_content(out.dump(2)));
            return { {"content", content} };
        }
        if (name == "delete_song") {
            const bool all = json_value(args, "all", false);
            std::vector<std::string> refs;
            if (args.contains("songs") && args.at("songs").is_array()) {
                for (const auto & v : args.at("songs")) {
                    if (v.is_string()) {
                        refs.push_back(v.get<std::string>());
                    }
                }
            } else if (args.contains("songs") && args.at("songs").is_string()) {
                refs.push_back(args.at("songs").get<std::string>());
            } else if (args.contains("song") && args.at("song").is_string()) {
                refs.push_back(args.at("song").get<std::string>());
            }
            if (!all && refs.empty()) {
                return tool_error("name the songs to delete (songs: [ids]) or pass all: true");
            }
            json deleted   = json::array();
            json files     = json::array();
            json not_found = json::array();
            {
                std::lock_guard<std::mutex> lk(mtx);
                std::vector<music_song_ptr> victims;
                if (all) {
                    victims = songs;
                } else {
                    for (const auto & r : refs) {
                        const music_song_ptr s = find_song(r);
                        if (!s) {
                            not_found.push_back(r);
                        } else if (std::find(victims.begin(), victims.end(), s) == victims.end()) {
                            victims.push_back(s);
                        }
                    }
                }
                for (const auto & s : victims) {
                    if (!s->file.empty()) {
                        std::error_code ec;
                        if (fs::remove(fs::u8path(s->file), ec)) {
                            files.push_back(s->file);
                        } else if (ec) {
                            SRV_WRN("MCP music: cannot delete %s: %s\n", s->file.c_str(), ec.message().c_str());
                        }
                    }
                    deleted.push_back(s->id);
                    songs.erase(std::remove(songs.begin(), songs.end(), s), songs.end());
                }
            }
            const json out = { {"deleted", deleted}, {"files_deleted", files}, {"not_found", not_found} };
            json content = json::array();
            content.push_back(text_content(out.dump(2)));
            return { {"content", content} };
        }
        if (name == "job_status") {
            const std::string id = json_value(args, "job_id", std::string());
            json out;
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (!id.empty()) {
                    const music_job_ptr j = find_job(id);
                    if (!j) {
                        return tool_error("no job \"" + id + "\"");
                    }
                    progress_report r = report(*j);
                    out = r.state;
                    out["message"] = r.text;
                } else {
                    json list = json::array();
                    for (auto it = jobs.rbegin(); it != jobs.rend() && list.size() < 10; ++it) {
                        progress_report r = report(**it);
                        json e = r.state;
                        e["message"] = r.text;
                        list.push_back(e);
                    }
                    out = { {"jobs", list}, {"queued", (int64_t) queue.size()} };
                }
            }
            json content = json::array();
            content.push_back(text_content(out.dump(2)));
            return { {"content", content} };
        }
        if (name == "cancel_job") {
            const std::string id = json_value(args, "job_id", std::string());
            std::string msg;
            {
                std::lock_guard<std::mutex> lk(mtx);
                music_job_ptr j;
                if (!id.empty()) {
                    j = find_job(id);
                    if (!j) {
                        return tool_error("no job \"" + id + "\"");
                    }
                } else {
                    for (const auto & x : jobs) {
                        if (x->status == "running") {
                            j = x;
                        }
                    }
                    if (!j && !queue.empty()) {
                        j = queue.back();
                    }
                    if (!j) {
                        return tool_error("no job is queued or running");
                    }
                }
                if (j->finished()) {
                    return tool_error(j->id + " already " + j->status);
                }
                j->cancel = true;
                const auto it = std::find(queue.begin(), queue.end(), j);
                if (it != queue.end()) {
                    queue.erase(it);
                    j->status     = "cancelled";
                    j->t_finished = now_seconds();
                    msg = j->id + " removed from the queue";
                } else {
                    msg = j->id + " stops after its current frame";
                }
            }
            cv.notify_all();
            json content = json::array();
            content.push_back(text_content(msg));
            return { {"content", content} };
        }
        if (name == "get_server_info") {
            json info = routes.tts_info();
            {
                std::lock_guard<std::mutex> lk(mtx);
                info["frames_per_second"] = k_frames_per_second;
                info["default_seconds"]   = k_default_seconds;
                info["min_seconds"]       = k_min_seconds;
                info["max_seconds"]       = k_max_seconds;
                info["queued"]            = (int64_t) queue.size();
                info["songs"]             = (int64_t) songs.size();
                info["running"]           = nullptr;
                for (const auto & j : jobs) {
                    if (j->status == "running") {
                        info["running"] = j->id;
                    }
                }
                info["save_dir"]     = dir.empty() ? json(nullptr) : json(dir.u8string());
                info["input_format"] = "caption (style) + lyrics with [verse]/[chorus]/... tags; the server builds the model prompt";
            }
            json content = json::array();
            content.push_back(text_content(info.dump(2)));
            return { {"content", content} };
        }
        return json(nullptr);  // unknown tool
    }

    // blocks until the job is finished or should_stop; true if finished
    bool wait_job(const music_job_ptr & j, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mtx);
        while (!j->finished()) {
            if (should_stop() || stopping) {
                return false;
            }
            cv.wait_for(lk, std::chrono::seconds(1));
        }
        return true;
    }
};

//
// HTTP
//

static server_http_res_ptr json_response(int status, const std::string & body) {
    auto res = std::make_unique<server_http_res>();
    res->status       = status;
    res->content_type = "application/json; charset=utf-8";
    res->data         = body;
    res->headers["Access-Control-Expose-Headers"] = "Mcp-Session-Id";
    return res;
}

static std::string base_url_of(const server_http_req & req) {
    std::string host  = header_value(req.headers, "Host");
    std::string proto = header_value(req.headers, "X-Forwarded-Proto");
    if (proto.empty()) {
        proto = "http";
    }
    if (host.empty()) {
        host = "localhost";
    }
    return proto + "://" + host;
}

server_mcp_music::server_mcp_music(const common_params & params, server_routes & routes)
    : pimpl(std::make_unique<impl>(params, routes)) {}

server_mcp_music::~server_mcp_music() = default;

void server_mcp_music::register_routes(const server_http_context & ctx_http) {
    impl * m = pimpl.get();

    ctx_http.post("/mcp", [m](const server_http_req & req) -> server_http_res_ptr {
        const json message = json::parse_no_throw(req.body);
        if (message.is_discarded()) {
            return json_response(400, rpc_error(nullptr, k_rpc_parse_error, "parse error").dump());
        }
        const std::string base_url    = base_url_of(req);
        const bool        accepts_sse = header_value(req.headers, "Accept").find("text/event-stream") != std::string::npos;

        // one request; a batch (array) is answered request by request, without streaming
        auto dispatch = [&](const json & msg, bool allow_stream, server_http_res_ptr * stream_res) -> json {
            if (!msg.is_object() || !msg.contains("method") || !msg.at("method").is_string()) {
                return rpc_error(msg.is_object() && msg.contains("id") ? msg.at("id") : json(nullptr), k_rpc_invalid_request, "invalid request");
            }
            const std::string method = msg.at("method").get<std::string>();
            const json        id     = msg.contains("id") ? msg.at("id") : json(nullptr);
            const json        params = msg.contains("params") && msg.at("params").is_object() ? msg.at("params") : json::object();
            if (!msg.contains("id")) {
                return json(nullptr);  // notification: nothing to answer
            }
            if (method == "initialize") {
                return rpc_result(id, m->initialize_result(params, base_url));
            }
            if (method == "ping") {
                return rpc_result(id, json::object());
            }
            if (method == "tools/list") {
                return rpc_result(id, m->tools_list());
            }
            if (method == "resources/list") {
                return rpc_result(id, { {"resources", json::array()} });
            }
            if (method == "resources/templates/list") {
                return rpc_result(id, { {"resourceTemplates", json::array()} });
            }
            if (method == "prompts/list") {
                return rpc_result(id, { {"prompts", json::array()} });
            }
            if (method != "tools/call") {
                return rpc_error(id, k_rpc_method_not_found, "method not found: " + method);
            }
            if (!params.contains("name") || !params.at("name").is_string()) {
                return rpc_error(id, k_rpc_invalid_params, "tools/call needs a tool name");
            }
            const std::string name = params.at("name").get<std::string>();
            const json args = params.contains("arguments") && params.at("arguments").is_object() ? params.at("arguments") : json::object();

            if (name != "generate_song") {
                const json result = m->call_tool(name, args, base_url);
                if (result.is_null()) {
                    return rpc_error(id, k_rpc_invalid_params, "unknown tool: " + name);
                }
                return rpc_result(id, result);
            }

            std::string error;
            const music_job_ptr j = m->queue_song(args, error);
            if (!j) {
                return rpc_result(id, tool_error(error));
            }
            const bool with_audio = json_value(args, "return_audio", true);

            if (!allow_stream) {
                if (!m->wait_job(j, req.should_stop)) {
                    return rpc_result(id, tool_error(j->id + " is still running: job_status reports it, get_latest_song fetches the song"));
                }
                std::lock_guard<std::mutex> lk(m->mtx);
                return rpc_result(id, m->job_result(*j, base_url, with_audio));
            }

            // stream: progress notifications (with a progressToken) or keep-alive comments, then the result
            json token = nullptr;
            if (params.contains("_meta") && params.at("_meta").is_object() && params.at("_meta").contains("progressToken")) {
                token = params.at("_meta").at("progressToken");
            }
            auto res = std::make_unique<server_http_res>();
            res->status       = 200;
            res->content_type = "text/event-stream";
            res->headers["Cache-Control"]                 = "no-cache";
            res->headers["Access-Control-Expose-Headers"] = "Mcp-Session-Id";
            struct stream_state {
                double last_progress  = 0;
                double last_keepalive = 0;
                double t0             = now_seconds();
                bool   first          = true;
                bool   ended          = false;
            };
            auto st = std::make_shared<stream_state>();
            res->next = [m, j, id, token, base_url, with_audio, st](std::string & output) -> bool {
                if (st->ended) {
                    return false;
                }
                std::unique_lock<std::mutex> lk(m->mtx);
                if (!st->first && !j->finished() && !m->stopping) {
                    m->cv.wait_for(lk, std::chrono::seconds(1));
                }
                st->first = false;
                if (j->finished() || m->stopping) {
                    const json result = j->finished() ? m->job_result(*j, base_url, with_audio)
                                                      : tool_error("the server is shutting down");
                    lk.unlock();
                    output    = sse_message(rpc_result(id, result));
                    st->ended = true;
                    return false;
                }
                const double now = now_seconds();
                if (token.is_null()) {
                    if (now - st->last_keepalive >= 10.0) {
                        output = ": waiting\n\n";
                        st->last_keepalive = now;
                    }
                    return true;
                }
                const impl::progress_report r = m->report(*j);
                // the spec wants the progress to rise with every notification
                const double progress = std::max(r.percent, st->last_progress + 0.001);
                st->last_progress = progress;
                output = sse_message({
                    {"jsonrpc", "2.0"},
                    {"method",  "notifications/progress"},
                    {"params",  { {"progressToken", token}, {"progress", progress}, {"total", 100}, {"message", r.text},
                                  {"_meta", { {"sdcpp/progress", r.state} }} }},
                });
                return true;
            };
            *stream_res = std::move(res);
            return json(nullptr);
        };

        if (message.is_array()) {
            json responses = json::array();
            for (const auto & item : message) {
                const json r = dispatch(item, false, nullptr);
                if (!r.is_null()) {
                    responses.push_back(r);
                }
            }
            if (responses.empty()) {
                return json_response(202, "");
            }
            return json_response(200, responses.dump());
        }

        server_http_res_ptr stream_res;
        const json r = dispatch(message, accepts_sse, &stream_res);
        if (stream_res) {
            return stream_res;
        }
        if (r.is_null()) {
            return json_response(202, "");
        }
        auto res = json_response(200, r.dump());
        if (message.is_object() && message.contains("method") && message.at("method").is_string() &&
            message.at("method").get<std::string>() == "initialize") {
            std::random_device rd;
            char sid[40];
            snprintf(sid, sizeof(sid), "%08x%08x%08x%08x", rd(), rd(), rd(), rd());
            res->headers["Mcp-Session-Id"] = sid;
        }
        return res;
    });

    // no server-initiated stream: every answer comes on the POST
    ctx_http.get("/mcp", [](const server_http_req &) -> server_http_res_ptr {
        auto res = json_response(405, rpc_error(nullptr, k_rpc_invalid_request, "use POST").dump());
        res->headers["Allow"] = "POST, DELETE";
        return res;
    });

    // the client ends its session; nothing is kept per session
    ctx_http.del("/mcp", [](const server_http_req &) -> server_http_res_ptr {
        return json_response(200, "");
    });

    ctx_http.get("/mcp/songs/:file", [m](const server_http_req & req) -> server_http_res_ptr {
        music_song_ptr s;
        {
            std::lock_guard<std::mutex> lk(m->mtx);
            s = m->find_song(req.get_param("file"));
        }
        std::shared_ptr<const std::string> wav = s ? impl::load_wav(s) : nullptr;
        if (!wav) {
            return json_response(404, json({ {"error", { {"code", 404}, {"message", "song not found"}, {"type", "not_found_error"} }} }).dump());
        }
        auto res = std::make_unique<server_http_res>();
        res->status       = 200;
        res->content_type = "audio/wav";
        res->data         = *wav;
        res->headers["Content-Disposition"] = "inline; filename=\"" + s->id + "_" + file_slug(s->title) + ".wav\"";
        res->headers["Cache-Control"]       = "private, max-age=86400";
        return res;
    });

    ctx_http.get("/mcp/songs", [m](const server_http_req & req) -> server_http_res_ptr {
        const std::string base_url = base_url_of(req);
        json list = json::array();
        {
            std::lock_guard<std::mutex> lk(m->mtx);
            for (auto it = m->songs.rbegin(); it != m->songs.rend(); ++it) {
                list.push_back(impl::describe(**it, base_url, false));
            }
        }
        return json_response(200, json({ {"songs", list} }).dump());
    });

    SRV_INF("MCP music endpoint at /mcp (songs at /mcp/songs)%s%s\n", m->dir.empty() ? "" : ", saving to ",
            m->dir.empty() ? "" : m->dir.u8string().c_str());
}
