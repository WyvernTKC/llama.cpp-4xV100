#pragma once

#include "server-http.h"

#include <memory>

struct common_params;
struct server_routes;

// MCP server endpoint for song generation (--enable-mcp-music): POST /mcp speaks MCP over streamable HTTP and offers
// generate_song, get_song, get_latest_song, list_songs, delete_song, job_status, cancel_job and get_server_info. Songs
// render one at a time on a worker thread through server_routes::run_tts; a tools/call that accepts text/event-stream
// gets MCP progress notifications while its song renders. GET /mcp/songs/<id>.wav serves the songs.
struct server_mcp_music {
    server_mcp_music(const common_params & params, server_routes & routes);
    ~server_mcp_music();

    void register_routes(const server_http_context & ctx_http);

    struct impl;
private:
    std::unique_ptr<impl> pimpl;
};
