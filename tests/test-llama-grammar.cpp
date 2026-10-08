#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"

#include "../src/llama-grammar.h"

#include <bitset>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main()
{
    llama_grammar_parser parsed_grammar;

    std::vector<std::pair<std::string, uint32_t>> expected = {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"ident", 8},
        {"ident_10", 10},
        {"num", 9},
        {"num_11", 11},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"term", 4},
        {"ws", 3},
        {"ws_12", 12},
    };

    std::vector<std::vector<llama_grammar_element>> expected_rules = {
        {{LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_CHAR, 10},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 4}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_END, 0}},
        {{LLAMA_GRETYPE_RULE_REF, 12}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 8},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_RULE_REF, 9},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 40},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 41},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 45},
            {LLAMA_GRETYPE_CHAR_ALT, 43},
            {LLAMA_GRETYPE_CHAR_ALT, 42},
            {LLAMA_GRETYPE_CHAR_ALT, 47},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 6}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 11}, {LLAMA_GRETYPE_RULE_REF, 3}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_CHAR_ALT, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_CHAR_ALT, 95},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_RULE_REF, 11},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 32},
            {LLAMA_GRETYPE_CHAR_ALT, 9},
            {LLAMA_GRETYPE_CHAR_ALT, 10},
            {LLAMA_GRETYPE_RULE_REF, 12},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
    };

    for (auto pair : expected)
    {
        parsed_grammar.symbol_ids[pair.first] = pair.second;
    }

    for (auto rule : expected_rules)
    {
        parsed_grammar.rules.emplace_back();
        for (auto element : rule)
        {
            parsed_grammar.rules.back().push_back(element);
        }
    }

    std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());

    llama_grammar * grammar = llama_grammar_init_impl(nullptr, grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    if (grammar == nullptr) {
        throw std::runtime_error("Failed to initialize llama_grammar");
    }

    std::vector<std::vector<llama_grammar_element>> expected_stacks = {
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        }};

    auto index = 0;
    for (const llama_grammar_stack & stack : llama_grammar_get_stacks(grammar))
    {
        // compare stack to expected_stack
        for (uint32_t i = 0; i < stack.size(); i++)
        {
            const llama_grammar_element * element = stack[i];
            const llama_grammar_element & expected_element = expected_stacks[index][i];

            // pretty print error message before asserting
            if (expected_element.type != element->type || expected_element.value != element->value)
            {
                fprintf(stderr, "index: %d\n", index);
                fprintf(stderr, "expected_element: %d, %u\n", expected_element.type, expected_element.value);
                fprintf(stderr, "actual_element: %d, %u\n", element->type, element->value);
                fprintf(stderr, "expected_element != actual_element\n");
            }

            assert(expected_element.type == element->type && expected_element.value == element->value);
        }
        index++;
    }

    std::vector<llama_grammar_candidate> next_candidates;
    next_candidates.resize(23);

    for (size_t i = 0; i < 23; ++i)
    {
        uint32_t *cp = new uint32_t[2]; // dynamically allocate memory for code_point
        cp[0] = 37 + i;
        cp[1] = 0;
        next_candidates[i] = {i, cp, {}, 0};
    }

    std::vector<std::vector<std::pair<uint32_t, uint16_t>>> expected_reject = {
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
        },
    };

    std::vector<llama_grammar_candidate> rejects = llama_grammar_reject_candidates_for_stack(llama_grammar_get_rules(grammar), llama_grammar_get_stacks(grammar)[0], next_candidates);

    std::vector<std::vector<llama_grammar_candidate>> all_rejects;

    for (std::size_t count = 0; count < llama_grammar_get_stacks(grammar).size(); ++count)
    {
        rejects = llama_grammar_reject_candidates_for_stack(llama_grammar_get_rules(grammar), llama_grammar_get_stacks(grammar)[count], next_candidates);
        all_rejects.push_back(rejects);
    }

    index = 0;
    for (auto rej : all_rejects)
    {
        for (uint32_t i = 0; i < rej.size(); i++)
        {
            auto element = rej[i];
            auto expected_element = expected_reject[index][i];
            assert(element.index == expected_element.first && *element.code_points == expected_element.second);
        }
        index++;
    }

    for (auto &candidate : next_candidates)
    {
        delete[] candidate.code_points;
        candidate.code_points = nullptr;
    }

    llama_grammar_free_impl(grammar);

    // The trigger patterns of a lazy grammar: a word escaped by regex_escape is searched as text, from where a match
    //   could end in the piece just appended; a pattern that says which bytes a match starts with is matched at those
    //   offsets only; everything else with the regex over the whole buffer. All find the leftmost match of the regex.
    //   Pieces are appended one at a time as llama_grammar_accept_impl does, and the buffer starts over at a match.
    {
        const char * patterns[] = {
            "<\\|tool_call>",          // "<|tool_call>" escaped
            "\\[TOOL_CALLS\\]",       // "[TOOL_CALLS]" escaped
            "<tool_call>",              // nothing to escape
            "[\\s\\S]*?(<tool_call>)", // starts with a class: the regex path
            "^\\s*\\{\"name\"",        // a line start
            // muse-glimmer: a line start or '<', lookaheads after it
            "(?:^|<\\|start\\|>assistant)( to=(?!self<\\|message\\|>)(?!user<\\|message\\|>)[^<]*?<\\|message\\|>)",
            "<\\|start\\|>assistant(\\s+to)", // gpt-oss
            "(?:x?(?:\"|y+?)|z*)\\{",      // optional terms before the first byte that must be there
            "(?:a|b*)c?[d]",            // a class after optional terms: the regex path
            "(?:T|m|t|@|<|\\||_|c|>|\\{)(?:o|h)", // many bytes: every offset looked at
            "(?:^|\\{)\\{",              // '{' after a line end found twice
        };
        const size_t n_patterns = sizeof(patterns)/sizeof(patterns[0]);
        llama_grammar * lazy = llama_grammar_init_impl(nullptr, "root ::= \"a\"", "root", true, patterns, n_patterns, nullptr, 0);
        assert(lazy != nullptr);
        assert(lazy->trigger_patterns.size() == n_patterns);
        assert(lazy->trigger_patterns[0].literal == "<|tool_call>");
        assert(lazy->trigger_patterns[1].literal == "[TOOL_CALLS]");
        assert(lazy->trigger_patterns[2].literal == "<tool_call>");
        for (size_t ip = 3; ip < n_patterns; ++ip) {
            assert(lazy->trigger_patterns[ip].literal.empty());
        }
        const auto first = [&](size_t ip, const std::string & bytes, const std::string & bol_bytes) {
            const llama_grammar_trigger_pattern & pat = lazy->trigger_patterns[ip];
            std::bitset<256> expected;
            std::bitset<256> expected_bol;
            for (const char c : bytes) {
                expected.set((unsigned char) c);
            }
            for (const char c : bol_bytes) {
                expected_bol.set((unsigned char) c);
            }
            assert(pat.first_known && !pat.first_bol_any && pat.first_bytes == expected && pat.first_bol_bytes == expected_bol);
        };
        assert(!lazy->trigger_patterns[3].first_known);
        first(4, "", std::string("\t\n\v\f\r {") + (char) 0x85 + (char) 0xa0);
        first(5, "<", " ");
        first(6, "<", "");
        first(7, "x\"yz{", "");
        assert(!lazy->trigger_patterns[8].first_known);
        first(9, "Tmt@<|_c>{", "");
        first(10, "{", "{");

        const std::vector<std::string> pieces = {
            "The ", "model ", "thinks", ". ", "<", "|", "tool", "_call", ">", " {\"name\"", "[TOOL", "_CALLS", "]",
            "<tool_", "call>", "<|tool_c", "all>", "tool_call>", "<", "<|", "\n", " ", "{", "\"name\"", "x",
            "<|start|>", "assistant", " to=", "self", "user", "functions.f", "<|message|>", "<|eom|>", "\r\n", "\t", "y",
            "z", "\"", "a", "b", "c", " to=user<|message|>", "\n to=f<|message|>", "\n{",
        };
        size_t n_checked = 0;
        size_t n_matched = 0;
        for (size_t ip = 0; ip < n_patterns; ++ip) {
            const llama_grammar_trigger_pattern & pat = lazy->trigger_patterns[ip];
            llama_grammar_trigger_pattern by_regex = pat;
            by_regex.literal.clear();
            by_regex.first_known = false;
            uint32_t rng = 12345 + (uint32_t) ip;
            std::string buffer;
            for (int step = 0; step < 1500; ++step) {
                rng = rng*1664525u + 1013904223u;
                const size_t searched = buffer.size();
                buffer += pieces[(rng >> 8) % pieces.size()];
                const size_t expected = by_regex.find(buffer);
                const size_t got      = pat.find(buffer, searched);
                if (expected != got) {
                    fprintf(stderr, "trigger pattern %zu step %d: expected %zu, got %zu in '%s'\n", ip, step, expected, got, buffer.c_str());
                    assert(false);
                }
                n_checked++;
                if (expected != std::string::npos) {
                    n_matched++;
                    buffer.clear();
                } else if (buffer.size() > 256) {
                    buffer.clear(); // a new grammar: the regex the result is checked against is slow on long text
                }
            }
        }
        assert(n_matched >= 20);

        // a match at the start of the input through the ^ alone
        for (const auto & [ip, input] : std::vector<std::pair<size_t, std::string>> {
                 { 4, " {\"name\"" }, { 5, " to=functions.f<|message|>" }, { 5, "x\n to=functions.f<|message|>" } }) {
            llama_grammar_trigger_pattern by_regex = lazy->trigger_patterns[ip];
            by_regex.first_known = false;
            const size_t expected = by_regex.find(input);
            assert(expected != std::string::npos);
            assert(lazy->trigger_patterns[ip].find(input) == expected);
        }

        // past a budget of anchored tries the search takes the rest
        {
            const llama_grammar_trigger_pattern & pat = lazy->trigger_patterns[5];
            llama_grammar_trigger_pattern by_regex = pat;
            by_regex.first_known = false;
            std::string buffer;
            for (int k = 0; k < 64; ++k) {
                buffer += "a <b ";
            }
            assert(pat.find(buffer) == std::string::npos && by_regex.find(buffer) == std::string::npos);
            const std::string call = "<|start|>assistant to=functions.f<|message|>";
            buffer += call;
            const size_t expected = buffer.size() - call.size() + std::string("<|start|>assistant").size();
            assert(by_regex.find(buffer) == expected);
            assert(pat.find(buffer) == expected);
        }
        fprintf(stderr, "trigger patterns: %zu searches, %zu matches, the regex search agrees\n", n_checked, n_matched);
        llama_grammar_free_impl(lazy);
    }

    return 0;
}
