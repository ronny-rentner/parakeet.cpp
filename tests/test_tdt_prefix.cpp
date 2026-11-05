#include "audio_io.hpp"
#include "encoder.hpp"
#include "ggml_graph.hpp"
#include "joint.hpp"
#include "mel.hpp"
#include "model_loader.hpp"
#include "phrase_boost.hpp"
#include "prediction.hpp"
#include "tdt.hpp"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

// A fixed token prefix must resume decoding at the matching encoder frame,
// returning exactly the remaining tokens. This checks state initialization,
// not recognition quality when a new clip is encoded separately.
int main() {
    const char* model_path = std::getenv("PARAKEET_TEST_GGUF");
    const char* audio_path = std::getenv("PARAKEET_TEST_PREFIX_WAV");
    const char* boost_json = std::getenv("PARAKEET_TEST_BOOST_PHRASES");
    if (!model_path || !audio_path || !boost_json) {
        std::fprintf(stderr, "test_tdt_prefix: model, audio and configured phrase strengths required; skip\n");
        return 77;
    }
    pk::set_num_threads(2);
    pk::ModelLoader loader;
    if (!loader.load(model_path)) return 1;
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(audio_path, audio)) return 1;
    pk::MelFrontend mel(loader);
    std::vector<float> feats;
    int n_mels, T;
    mel.compute(audio.samples, feats, n_mels, T);
    pk::Encoder encoder(loader);
    std::vector<float> channels_first;
    int H, frames;
    encoder.forward(feats, n_mels, T, channels_first, H, frames);
    std::vector<float> enc(channels_first.size());
    for (int t = 0; t < frames; ++t)
        for (int h = 0; h < H; ++h)
            enc[t * H + h] = channels_first[h * frames + t];
    std::vector<std::pair<std::string, float>> phrases;
    const auto configured_boosts = nlohmann::json::parse(boost_json);
    for (const auto& item : configured_boosts.items())
        phrases.emplace_back(item.key(), item.value().get<float>());
    pk::PhraseBoost boost(loader.tokenizer_pieces(), phrases);
    pk::PredictionNet pred(loader);
    pk::Joint joint(loader);
    const auto& cfg = loader.config();
    std::vector<pk::TokenInfo> full_tokens;
    const auto full = pk::tdt_greedy(pred, joint, enc, frames, H, cfg.tdt_durations, cfg.blank_id, cfg.max_symbols, &full_tokens, &boost);
    int checked = 0;
    for (size_t split = 1; split < full_tokens.size(); ++split) {
        if (full_tokens[split].frame == full_tokens[split - 1].frame) continue;
        const int frame = full_tokens[split].frame;
        const std::vector<int32_t> prefix(full.begin(), full.begin() + split);
        const std::vector<float> tail(enc.begin() + frame * H, enc.end());
        const auto continued = pk::tdt_greedy(pred, joint, tail, frames - frame, H, cfg.tdt_durations, cfg.blank_id, cfg.max_symbols, nullptr, &boost, prefix);
        const std::vector<int32_t> expected(full.begin() + split, full.end());
        if (continued != expected) {
            std::fprintf(stderr, "test_tdt_prefix: continuation differs after token %zu\n", split);
            return 1;
        }
        ++checked;
    }
    const auto fresh = pk::tdt_greedy(pred, joint, enc, frames, H, cfg.tdt_durations, cfg.blank_id, cfg.max_symbols, nullptr, &boost);
    if (fresh != full || checked == 0) return 1;
    std::printf("test_tdt_prefix: %d continuation boundaries and fresh-request isolation passed\n", checked);
    pk::shutdown_backend();
    return 0;
}
