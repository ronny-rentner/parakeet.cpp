#include "spelling.hpp"

#include <cstdio>

static int fails = 0;
static void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++fails; }
}

int main() {
    using pkserver::correct_spelling;
    const std::vector<std::pair<std::string, std::string>> spelling = {
        {"Ghostty", "ghostty"}, {"Ghostty", "ghost ty"}, {"Ghostty", "ghost-ty"}, {"TOML", "toml"},
    };
    check(correct_spelling("GhostTY GhostTy GHOSTTY", spelling) == "Ghostty Ghostty Ghostty", "ASCII case variants");
    check(correct_spelling("ghost ty, ghost-ty; toml!", spelling) == "Ghostty, Ghostty; TOML!", "explicit spaces and dashes");
    check(correct_spelling("ghosttys xghostty ghostty2 ghostty_name äghostty ghosttyé", spelling) == "ghosttys xghostty ghostty2 ghostty_name äghostty ghosttyé",
          "do not replace inside words");
    check(correct_spelling("(GHOSTTY), „GhostTY“—toml!", spelling) == "(Ghostty), „Ghostty“—TOML!", "preserve surrounding punctuation");
    check(correct_spelling("GHOSTTY ghost-ty ghost  ty ghost\nty ghost ty", {{"Ghostty", "ghost ty"}}) == "GHOSTTY ghost-ty ghost  ty ghost\nty Ghostty",
          "only explicit alternatives match, with literal spacing");
    check(correct_spelling("VSCODE", {{"Visual Studio Code", "vscode"}}) == "Visual Studio Code", "one word becomes several");
    check(correct_spelling("MÜNCHEN münchen ÖL öl ÄPFEL äpfel", {{"München", "münchen"}, {"Öl", "öl"}, {"Äpfel", "äpfel"}}) == "München München Öl Öl Äpfel Äpfel",
          "shared lowercase handles all three umlaut pairs");
    check(correct_spelling("münchen", {{"München", "MÜNCHEN"}}) == "München", "normalize alternatives as well as transcript");
    check(correct_spelling("STRAßE STRAẞE STRASSE", {{"Straße", "straße"}}) == "Straße STRAẞE STRASSE", "no sharp-s conversion");
    check(correct_spelling("É é", {{"accent", "é"}}) == "É accent", "other non-ASCII letters match exactly");
    check(correct_spelling("git hub", {{"Git", "git"}, {"GitHub", "git hub"}}) == "GitHub", "longest alternative wins");
    check(correct_spelling("GITHUB", {{"GitHub", "github"}, {"Other", "GitHub"}}) == "GitHub", "first duplicate wins");
    check(correct_spelling("ghost ty", {{"Ghostty", "ghost ty"}, {"Other", "ghostty"}}) == "Ghostty", "no cascading replacements");
    check(correct_spelling("QUOTE", {{"a\"b\\c", "quote"}}) == "a\"b\\c", "output preserves quotes and backslashes literally");
    check(correct_spelling("GHOSTTY", {}) == "GHOSTTY", "absent mappings leave text unchanged");
    check(correct_spelling("", spelling).empty(), "empty transcript");
    if (fails) { std::fprintf(stderr, "%d checks failed\n", fails); return 1; }
    std::printf("test_server_spelling: OK\n");
    return 0;
}
