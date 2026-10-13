#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pk {

// Match decoded pieces so every tokenization of a phrase receives the same bonus.
class PhraseBoost {
public:
    PhraseBoost(const std::vector<std::string>& pieces,
                const std::vector<std::pair<std::string, float>>& phrases) {
        struct Node {
            std::map<unsigned char, int> edges;
            int fail = 0;
            float potential = 0;
            float complete = 0;
            bool space = false;
        };
        std::vector<Node> nodes(2);
        nodes[0].edges[' '] = 1;
        nodes[1].space = true;
        for (const auto& [phrase, score] : phrases) {
            if (!std::isfinite(score) || score < 0)
                throw std::invalid_argument("phrase strength must be finite and non-negative");
            std::string text = normalize(phrase);
            const size_t begin = text.find_first_not_of(' ');
            if (text.find_first_not_of(" .,!?;:\"()[]{}") == std::string::npos)
                throw std::invalid_argument("boost_phrase must contain a word");
            text = text.substr(begin, text.find_last_not_of(' ') - begin + 1);
            const int length = std::count_if(text.begin(), text.end(), [](unsigned char c) {
                return c != ' ' && (c & 0xc0) != 0x80;
            });
            int state = 1, consumed = 0;
            for (unsigned char c : text) {
                auto edge = nodes[state].edges.find(c);
                if (edge == nodes[state].edges.end()) {
                    const int next = (int)nodes.size();
                    nodes[state].edges[c] = next;
                    nodes.emplace_back();
                    nodes.back().space = c == ' ';
                    state = next;
                } else {
                    state = edge->second;
                }
                if (c != ' ' && (c & 0xc0) != 0x80) ++consumed;
                nodes[state].potential = std::max(nodes[state].potential, score * consumed / length);
            }
            nodes[state].complete = std::max(nodes[state].complete, score);
        }

        std::queue<int> pending;
        pending.push(1);
        while (!pending.empty()) {
            const int state = pending.front();
            pending.pop();
            for (const auto& edge : nodes[state].edges) {
                int fallback = nodes[state].fail;
                while (fallback && !nodes[fallback].edges.count(edge.first))
                    fallback = nodes[fallback].fail;
                const auto next = nodes[fallback].edges.find(edge.first);
                if (next != nodes[fallback].edges.end()) fallback = next->second;
                nodes[edge.second].fail = fallback;
                nodes[edge.second].potential = std::max(nodes[edge.second].potential, nodes[fallback].potential);
                nodes[edge.second].complete = std::max(nodes[edge.second].complete, nodes[fallback].complete);
                pending.push(edge.second);
            }
        }

        std::vector<std::string> text_pieces;
        for (const std::string& piece : pieces) {
            const bool special = !piece.empty() &&
                ((piece.front() == '<' && piece.back() == '>') ||
                 (piece.front() == '[' && piece.back() == ']'));
            text_pieces.push_back(special ? "" : normalize(piece));
        }
        transitions_.resize(nodes.size());
        for (size_t start = 0; start < nodes.size(); ++start) {
            for (const std::string& piece : text_pieces) {
                int state = (int)start;
                float completed = 0;
                for (unsigned char c : piece) {
                    if (c == ' ' && nodes[state].space) continue;
                    if (c == ' ' || std::string(".,!?;:\"()[]{}").find(c) != std::string::npos) {
                        completed += nodes[state].complete;
                        if (c != ' ') { state = 1; continue; }
                    }
                    while (state && !nodes[state].edges.count(c)) state = nodes[state].fail;
                    const auto next = nodes[state].edges.find(c);
                    state = next == nodes[state].edges.end() ? 0 : next->second;
                }
                // A failed prefix loses its provisional bonus; only whole words keep it.
                transitions_[start].push_back({state, completed + nodes[state].potential - nodes[start].potential});
            }
        }
    }

    int initial_state() const { return 1; }

    void apply(std::vector<float>& logits, int state) const {
        for (size_t token = 0; token < transitions_[state].size(); ++token)
            logits[token] += transitions_[state][token].bonus;
    }

    int advance(int state, int token) const { return transitions_[state][token].state; }

private:
    static std::string normalize(const std::string& text) {
        std::string result;
        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char c = text[i];
            if (text.compare(i, 3, "▁") == 0) { c = ' '; i += 2; }
            if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
            if (std::string(" \t\r\n").find(c) != std::string::npos) c = ' ';
            if (c != ' ' || result.empty() || result.back() != ' ') result += (char)c;
        }
        return result;
    }

    struct Transition { int state; float bonus; };
    std::vector<std::vector<Transition>> transitions_;
};

} // namespace pk
