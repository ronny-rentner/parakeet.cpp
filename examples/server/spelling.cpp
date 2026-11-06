#include "spelling.hpp"
#include "common.hpp"

#include <cctype>
#include <string_view>

namespace pkserver {

std::string correct_spelling(const std::string& text, std::vector<std::pair<std::string, std::string>> spelling) {
    if (spelling.empty()) return text;
    for (auto& rule : spelling) rule.second = pk::lowercase(std::move(rule.second));
    // The shared lowercase function preserves byte lengths, so matches index the original text.
    const std::string folded = pk::lowercase(text);
    auto boundary = [&](size_t pos, bool before) {
        if ((before && pos == 0) || (!before && pos == text.size())) return true;
        const unsigned char c = text[before ? pos - 1 : pos];
        if (c < 0x80) return !std::isalnum(c) && c != '_';
        // Recognize common UTF-8 punctuation without treating bytes inside letters as separators.
        for (const std::string_view separator : {"\u00a0", "„", "“", "”", "‚", "‘", "’", "«", "»", "‹", "›", "–", "—", "…"}) {
            if (before && pos < separator.size()) continue;
            if (text.compare(before ? pos - separator.size() : pos, separator.size(), separator) == 0) return true;
        }
        return false;
    };
    std::string result;
    result.reserve(text.size());
    for (size_t pos = 0; pos < text.size();) {
        const std::pair<std::string, std::string>* match = nullptr;
        if (boundary(pos, true)) {
            for (const auto& rule : spelling) {
                if (match && rule.second.size() <= match->second.size()) continue;
                if (folded.compare(pos, rule.second.size(), rule.second) == 0 && boundary(pos + rule.second.size(), false)) match = &rule;
            }
        }
        // Use the longest match, keeping the first rule on ties; never rescan replacements.
        if (match) {
            result += match->first;
            pos += match->second.size();
        } else {
            result += text[pos++];
        }
    }
    return result;
}

} // namespace pkserver
