#pragma once
#include <string>
#include <utility>
#include <vector>

namespace pkserver {

// Mappings are output/alternative pairs; alternatives must be nonempty.
std::string correct_spelling(const std::string& text, std::vector<std::pair<std::string, std::string>> spelling);

} // namespace pkserver
