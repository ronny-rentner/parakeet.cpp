#include "phrase_boost.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

int main() {
    const std::vector<std::string> pieces{
        "▁sub", "mit", "▁Submit", ".", "ted", "▁clear", "▁line", "▁nine", "▁other",
        "▁resubmit", "▁sy", "mlink", "▁symlink", "▁café", "<unk>", "▁",
        "▁grammar", "▁spelling", "▁and"};
    const pk::PhraseBoost boost(pieces, {{"submit", 8}, {"clear line", 4}, {"symlink", 6}, {"café", 2},
                                         {"grammar", 2}, {"spelling and grammar", 6}, {"SUBMIT", 3}});
    auto check = [&](const char* label, const std::vector<int>& tokens, float expected) {
        int state = boost.initial_state();
        float total = 0;
        for (int token : tokens) {
            std::vector<float> logits(pieces.size() + 6, 0.0f);
            boost.apply(logits, state);
            for (size_t i = pieces.size(); i < logits.size(); ++i) {
                if (logits[i] != 0) {
                    std::fprintf(stderr, "blank/duration logit changed\n");
                    return false;
                }
            }
            total += logits[token];
            state = boost.advance(state, token);
        }
        if (std::fabs(total - expected) > 1e-5f) {
            std::fprintf(stderr, "%s: got %g, expected %g\n", label, total, expected);
            return false;
        }
        return true;
    };
    if (!check("split word", {0, 1, 3}, 8) ||
        !check("whole word, English case", {2, 3}, 8) ||
        !check("longer word must cancel", {2, 4, 3}, 0) ||
        !check("word inside another word", {9, 3}, 0) ||
        !check("incomplete prefix cancels", {0, 8, 3}, 0) ||
        !check("phrase", {5, 6, 3}, 4) ||
        !check("first word alone", {5, 3}, 0) ||
        !check("second word alone", {6, 3}, 0) ||
        !check("wrong phrase", {5, 7, 3}, 0) ||
        !check("reversed phrase", {6, 5, 3}, 0) ||
        !check("split technical term", {10, 11, 3}, 6) ||
        !check("whole technical term", {12, 3}, 6) ||
        !check("Unicode term", {13, 3}, 2) ||
        !check("repeat and reset", {2, 3, 8, 5, 6, 3}, 12) ||
        !check("word with phrase suffix", {16, 3}, 2) ||
        !check("overlapping phrase", {17, 18, 16, 3}, 6) ||
        !check("unfinished overlapping phrase", {17, 18, 8, 3}, 0) ||
        !check("special token and separate space", {15, 14, 0, 1, 3}, 8)) return 1;

    const pk::PhraseBoost zero(pieces, {{"submit", 0}});
    std::vector<float> logits(pieces.size() + 6, -2.0f);
    zero.apply(logits, zero.initial_state());
    for (float value : logits) if (value != -2.0f) return 1;
    try {
        const pk::PhraseBoost invalid(pieces, {{" . ", 2}});
        std::fprintf(stderr, "empty phrase accepted\n");
        return 1;
    } catch (const std::invalid_argument&) {}
    for (float score : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        try {
            const pk::PhraseBoost invalid(pieces, {{"submit", score}});
            std::fprintf(stderr, "invalid strength accepted\n");
            return 1;
        } catch (const std::invalid_argument&) {}
    }
    return 0;
}
