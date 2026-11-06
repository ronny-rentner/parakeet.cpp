#include "common.hpp"

#include <cstdio>

static int fails = 0;
static void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++fails; }
}

int main() {
    check(pk::lowercase("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == "abcdefghijklmnopqrstuvwxyz", "ASCII uppercase");
    check(pk::lowercase("abcdefghijklmnopqrstuvwxyz") == "abcdefghijklmnopqrstuvwxyz", "ASCII lowercase unchanged");
    check(pk::lowercase("ÄÖÜäöüÄÄ") == "äöüäöüää", "all umlaut pairs, including repeated occurrences");
    check(pk::lowercase("MÜNCHEN, ÖL und Äpfel!") == "münchen, öl und äpfel!", "mixed ASCII and umlauts");
    check(pk::lowercase("ß ẞ É é Σ σ 中 🙂") == "ß ẞ É é Σ σ 中 🙂", "other UTF-8 characters unchanged");
    check(pk::lowercase("  A-B\tC_42\r\n") == "  a-b\tc_42\r\n", "preserve whitespace, punctuation and digits");
    check(pk::lowercase(std::string("A\0B", 3)) == std::string("a\0b", 3), "preserve embedded null bytes");
    check(pk::lowercase("").empty(), "empty input");
    if (fails) { std::fprintf(stderr, "%d checks failed\n", fails); return 1; }
    std::printf("test_lowercase: OK\n");
    return 0;
}
