#include "../src/unicode.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static void print_split(const char * name, const std::vector<std::string> & split) {
    fprintf(stderr, "  %s:", name);
    for (const auto & piece : split) {
        fprintf(stderr, " [%s]", piece.c_str());
    }
    fprintf(stderr, "\n");
}

// The custom splitter of the GPT-4o (llama4, gpt-oss) regex against std::regex: the same regex inside a group is not
//   one the custom splitters know, so it goes the std::regex way. Random lines of letters of both cases (ASCII and
//   not), digits, contractions, punctuation, symbols, marks and spaces of all kinds; short ones, as std::regex runs out
//   of stack on long runs with MSVC.
static int test_gpt4o_split() {
    const std::string re = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const std::vector<std::string> pieces = {
        std::string("a"),
        std::string("b"),
        std::string("z"),
        std::string("A"),
        std::string("B"),
        std::string("Z"),
        std::string("\xc3" "\xa9"),
        std::string("\xc3" "\x89"),
        std::string("\xd0" "\xb6"),
        std::string("\xd0" "\x96"),
        std::string("\xe6" "\x97" "\xa5"),
        std::string("\xc3" "\x9f"),
        std::string("\xc7" "\x85"),
        std::string("0"),
        std::string("7"),
        std::string("\xd9" "\xa3"),
        std::string("\xe2" "\x85" "\xab"),
        std::string("'"),
        std::string("'s"),
        std::string("'S"),
        std::string("'t"),
        std::string("'re"),
        std::string("'RE"),
        std::string("'Ve"),
        std::string("'m"),
        std::string("'ll"),
        std::string("'LL"),
        std::string("'d"),
        std::string("'x"),
        std::string("'r"),
        std::string("'l"),
        std::string(" "),
        std::string("  "),
        std::string("\x09"),
        std::string("\x0a"),
        std::string("\x0d" "\x0a"),
        std::string("\x0d"),
        std::string("\xc2" "\xa0"),
        std::string("\xe3" "\x80" "\x80"),
        std::string("/"),
        std::string("-"),
        std::string("."),
        std::string(","),
        std::string("!"),
        std::string("$"),
        std::string("+"),
        std::string("\xe2" "\x82" "\xac"),
        std::string("\xc2" "\xa9"),
        std::string("\xcc" "\x81"),
        std::string("\xf0" "\x9f" "\x98" "\x80"),
        std::string("_"),
        std::string("\""),
        std::string("don"),
        std::string("DON"),
        std::string("McDonald"),
        std::string("HTTPServer"),
        std::string("123456"),
    };
    uint32_t rng = 12345;
    auto next = [&rng]() {
        rng = rng*1664525u + 1013904223u;
        return rng >> 8;
    };
    for (int line = 0; line < 4000; ++line) {
        std::string text;
        const int n = 1 + next() % 40;
        for (int k = 0; k < n; ++k) {
            text += pieces[next() % pieces.size()];
        }
        const auto custom = unicode_regex_split(text, { re }, false);
        const auto stl    = unicode_regex_split(text, { "(?:" + re + ")" }, false);
        if (custom != stl) {
            fprintf(stderr, "gpt4o split differs for '%s'\n", text.c_str());
            print_split("custom", custom);
            print_split("std::regex", stl);
            return 1;
        }
    }
    return 0;
}

int main() {
    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "unexpected split:");
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        return 1;
    }

    return test_gpt4o_split();
}
