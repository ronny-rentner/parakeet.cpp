#include "httplib.h"
#include "model_fetch.hpp"
#include "openai_format.hpp"

#include "parakeet.h"        // parakeet_version
#include "model.hpp"          // pk::Model
#include "transcription.hpp"  // pk::Transcription
#include "ggml_graph.hpp"     // pk::set_num_threads, pk::shutdown_backend
#include "dr_wav.h"           // declarations only; impl lives in libparakeet
#include "ced_tagger.hpp"     // pk::CedTagger
#include "sound_stream.hpp"   // pk::SoundStream, pk::SoundSegment
#include "audio_io.hpp"       // pk::resample_linear
#include "phrase_boost.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

httplib::Server* g_server = nullptr;
void on_signal(int) { if (g_server) g_server->stop(); }

// Decode WAV bytes in memory into mono float PCM. Returns false if the bytes
// are not a decodable, non-empty WAV (so a bad/empty upload becomes a 400, not
// a 500 from the inference path). The model resamples to 16 kHz internally, so
// the native sample rate is returned as-is.
bool decode_wav_mem(const std::string& bytes, std::vector<float>& mono,
                    int& sample_rate) {
    unsigned int ch = 0, sr = 0;
    drwav_uint64 frames = 0;
    float* pcm = drwav_open_memory_and_read_pcm_frames_f32(
        bytes.data(), bytes.size(), &ch, &sr, &frames, nullptr);
    if (!pcm || ch == 0 || sr == 0 || frames == 0) {
        if (pcm) drwav_free(pcm, nullptr);
        return false;
    }
    mono.resize(frames);
    for (drwav_uint64 i = 0; i < frames; ++i) {
        double acc = 0;
        for (unsigned int c = 0; c < ch; ++c) acc += pcm[i * ch + c];
        mono[i] = (float)(acc / ch);
    }
    drwav_free(pcm, nullptr);
    sample_rate = (int)sr;
    return true;
}

void usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  parakeet-server --model <path|url|alias> [--host 127.0.0.1] "
        "[--port 8080] [--threads N] [--concurrency K] [--model-dir <dir>] [--sound-model <path>] [--unload-after SEC] [--fetch-missing]\n"
        "  parakeet-server --fetch <url|alias|name.gguf> [--fetch ...] [--model-dir <dir>]\n"
        "\n"
        "--fetch downloads each model into the model directory and exits without serving.\n"
        "A --model given as URL, alias or name.gguf must already be in the model directory unless --fetch-missing is set.\n"
        "Serves POST /v1/audio/transcriptions (OpenAI-compatible). A request may name another GGUF\n"
        "from the --model directory; models unused for --unload-after seconds are unloaded (default 300, 0 keeps them).\n"
        "--concurrency K runs up to K transcriptions at once on K CPU backends "
        "(default 1); --threads is then the thread count of each backend, and "
        "K x threads should not exceed the physical cores.\n"
        "--sound-model loads a ced.cpp sound-event tagger (local GGUF path); "
        "verbose_json responses then include a \"sound_events\" array.\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string model_arg, host = "127.0.0.1", model_dir, sound_model;
    std::vector<std::string> fetch;
    int port = 8080, threads = 0, concurrency = 1, unload_after = 300;
    bool fetch_missing = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model")        model_arg = next("--model");
        else if (a == "--host")    host = next("--host");
        else if (a == "--port")    port = std::atoi(next("--port").c_str());
        else if (a == "--threads") threads = std::atoi(next("--threads").c_str());
        else if (a == "--concurrency") concurrency = std::atoi(next("--concurrency").c_str());
        else if (a == "--model-dir") model_dir = next("--model-dir");
        else if (a == "--sound-model") sound_model = next("--sound-model");
        else if (a == "--unload-after") unload_after = std::atoi(next("--unload-after").c_str());
        else if (a == "--fetch")   fetch.push_back(next("--fetch"));
        else if (a == "--fetch-missing") fetch_missing = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--version" || a == "-V") {
            std::printf("parakeet-server %s\n", parakeet_version());
            return 0;
        }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(); return 2; }
    }
    if (model_dir.empty()) model_dir = pkserver::default_model_dir();
    if (!fetch.empty()) {
        for (const std::string& name : fetch) {
            pkserver::ModelSource src;
            std::string err;
            if (!pkserver::resolve_model(name, src, err)) {
                std::fprintf(stderr, "parakeet-server: %s\n", err.c_str());
                return 2;
            }
            try {
                std::fprintf(stderr, "parakeet-server: %s\n", pkserver::fetch_model(src, model_dir).c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "parakeet-server: %s\n", e.what());
                return 1;
            }
        }
        return 0;
    }
    if (model_arg.empty()) { usage(); return 2; }

    // Resolve and fetch the model before binding the socket.
    pkserver::ModelSource src;
    std::string err;
    if (!pkserver::resolve_model(model_arg, src, err)) {
        std::fprintf(stderr, "parakeet-server: %s\n", err.c_str());
        return 2;
    }
    std::string model_path;
    try {
        if (src.kind == pkserver::ModelSource::kUrl)
            std::fprintf(stderr, "parakeet-server: fetching %s\n", src.value.c_str());
        model_path = pkserver::fetch_model(src, model_dir, fetch_missing);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "parakeet-server: %s\n", e.what());
        return 1;
    }

    if (concurrency < 1) concurrency = 1;
    // With a pool, --threads is the thread count of each backend and the global
    // override stays unset (pooled backends ignore it).
    if (threads > 0 && concurrency == 1) pk::set_num_threads(threads);

    std::unique_ptr<pk::Model> model = pk::Model::load(model_path);
    if (!model) {
        std::fprintf(stderr, "parakeet-server: failed to load model %s\n", model_path.c_str());
        pk::shutdown_backend();
        return 1;
    }
    if (concurrency > 1) {
        const int eff = model->set_concurrency(concurrency, threads);
        if (eff != concurrency)
            std::fprintf(stderr, "parakeet-server: --concurrency %d not available on this device; using %d\n",
                         concurrency, eff);
        concurrency = eff;
    }
    const bool pooled = concurrency > 1;
    // Loaded models stay until idle for --unload-after seconds; `last_used` is stamped at inference.
    struct LoadedModel {
        std::shared_ptr<pk::Model> model;
        std::chrono::steady_clock::time_point last_used;
    };
    std::map<std::string, LoadedModel> models;
    models.emplace(model_path, LoadedModel{std::move(model), std::chrono::steady_clock::now()});
    std::mutex models_mu;
    std::condition_variable models_cv;
    bool stopping = false;
    std::thread unloader;
    if (unload_after > 0) unloader = std::thread([&] {
        std::unique_lock<std::mutex> lock(models_mu);
        while (!models_cv.wait_for(lock, std::chrono::seconds(10), [&] { return stopping; })) {
            const auto cutoff = std::chrono::steady_clock::now() - std::chrono::seconds(unload_after);
            for (auto it = models.begin(); it != models.end();) {
                // A request holding its own shared_ptr keeps use_count above one while it runs.
                if (it->second.last_used < cutoff && it->second.model.use_count() == 1) {
                    std::fprintf(stderr, "parakeet-server: unloaded idle model %s\n", it->first.c_str());
                    it = models.erase(it);
                } else {
                    ++it;
                }
            }
        }
    });

    // --sound-model is a local path only; it does not go through the ASR
    // model's alias/URL resolver.
    std::unique_ptr<pk::CedTagger> tagger;
    if (!sound_model.empty()) {
        if (!pk::CedTagger::available()) {
            std::fprintf(stderr, "parakeet-server: built without sound tagging (PARAKEET_WITH_CED=OFF)\n");
            pk::shutdown_backend();
            return 2;
        }
        tagger = pk::CedTagger::load(sound_model);
        if (!tagger) {
            std::fprintf(stderr, "parakeet-server: failed to load sound model %s\n", sound_model.c_str());
            pk::shutdown_backend();
            return 1;
        }
    }

    std::mutex infer_mu;

    httplib::Server svr;
    g_server = &svr;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/v1/audio/transcriptions",
             [&](const httplib::Request& req, httplib::Response& res) {
        using namespace pkserver;
        auto fail = [&](int code, const std::string& msg) {
            res.status = code;
            res.set_content(error_body(msg, "invalid_request_error"), "application/json");
        };

        if (!req.has_file("file")) { fail(400, "missing required field 'file'"); return; }

        std::shared_ptr<pk::Model> model;
        std::string path = model_path;
        {
            std::lock_guard<std::mutex> lock(models_mu);
            if (req.has_file("model") && !req.get_file_value("model").content.empty()) {
                // A bare file name in the directory of the --model file; paths and URLs are refused.
                const std::string& requested = req.get_file_value("model").content;
                if (requested.find('/') != std::string::npos) {
                    fail(400, "model must be a file name, not a path: " + requested);
                    return;
                }
                path = (std::filesystem::path(model_path).parent_path() / requested).string();
                if (!std::filesystem::is_regular_file(path)) {
                    fail(400, "model file does not exist: " + path);
                    return;
                }
            }
            auto found = models.find(path);
            if (found == models.end()) {
                std::shared_ptr<pk::Model> loaded = pk::Model::load(path);
                if (!loaded) { fail(500, "could not load model file: " + path); return; }
                if (pooled) loaded->set_concurrency(concurrency, threads);
                found = models.emplace(path, LoadedModel{std::move(loaded), std::chrono::steady_clock::now()}).first;
                std::fprintf(stderr, "parakeet-server: loaded model %s\n", path.c_str());
            }
            model = found->second.model;
        }

        // response_format (default json)
        Format fmt = Format::kJson;
        if (req.has_file("response_format")) {
            const std::string& rf = req.get_file_value("response_format").content;
            if (!parse_format(rf, fmt)) { fail(400, "unsupported response_format '" + rf + "'"); return; }
        }

        // Every `boost` field holds lines of "<phrase> <strength>"; the strength is the last word.
        std::vector<std::pair<std::string, float>> boost_phrases;
        for (const auto& field : req.get_file_values("boost")) {
            std::istringstream lines(field.content);
            std::string line;
            while (std::getline(lines, line)) {
                const size_t end = line.find_last_not_of(" \t\r");
                if (end == std::string::npos) continue;
                const size_t split = line.find_last_of(" \t", end);
                if (split == std::string::npos) {
                    fail(400, "boost line needs a phrase and a strength: " + line);
                    return;
                }
                float score;
                const auto parsed = std::from_chars(line.data() + split + 1, line.data() + end + 1, score);
                if (parsed.ec != std::errc{} || parsed.ptr != line.data() + end + 1) {
                    fail(400, "boost strength must be a number: " + line);
                    return;
                }
                boost_phrases.emplace_back(line.substr(0, split), score);
            }
        }
        std::unique_ptr<pk::PhraseBoost> boost;
        if (!boost_phrases.empty()) {
            if (model->config().tdt_durations.empty() || fmt == Format::kVerboseJson) {
                fail(400, "phrase boosting requires a TDT model and response_format json or text");
                return;
            }
            try {
                boost = std::make_unique<pk::PhraseBoost>(model->loader().tokenizer_pieces(), boost_phrases);
            } catch (const std::invalid_argument& e) {
                fail(400, e.what());
                return;
            }
        }

        // timestamp_granularities[] = word -> include words[]
        bool include_words = false;
        for (const auto& g : req.get_file_values("timestamp_granularities[]"))
            if (g.content == "word") include_words = true;

        const std::string& bytes = req.get_file_value("file").content;
        std::vector<float> pcm;
        int sr = 0;
        if (!decode_wav_mem(bytes, pcm, sr)) {
            fail(400, "could not decode audio; this example accepts WAV uploads only");
            return;
        }
        double duration_sec = sr > 0 ? (double)pcm.size() / (double)sr : 0.0;

        try {
            pk::Transcription tr;
            std::vector<SoundEventOut> sound_events;
            bool have_sounds = false;
            {
                // One backend: hold the lock across ASR and tagging, as before.
                // With a pool, ASR runs in parallel on its own backend and only
                // the tagger (not thread-safe) takes the lock.
                std::unique_lock<std::mutex> lock(infer_mu, std::defer_lock);
                if (!pooled) lock.lock();
                {
                    std::lock_guard<std::mutex> lock(models_mu);
                    models.at(path).last_used = std::chrono::steady_clock::now();
                }
                tr = model->transcribe_with_timestamps(pcm, sr, pk::Decoder::kDefault, "", boost.get());

                // Sound tagging is verbose_json-only so json/text requests never
                // pay for it. Runs under the same lock as ASR: the tagger is not
                // thread-safe. A failure here must not fail the transcription.
                if (tagger && fmt == Format::kVerboseJson) {
                    if (pooled) lock.lock();
                    try {
                        const std::vector<float>* mono16k = &pcm;
                        std::vector<float> resampled;
                        if (sr != 16000) {
                            resampled = pk::resample_linear(pcm, sr, 16000);
                            mono16k = &resampled;
                        }
                        pk::SoundStream stream(tagger->scorer(), tagger->n_classes(), pk::SoundOpts{});
                        std::vector<pk::SoundSegment> segs =
                            stream.feed(mono16k->data(), (int)mono16k->size(), /*is_last=*/true);
                        sound_events.reserve(segs.size());
                        for (const pk::SoundSegment& s : segs) {
                            const char* label = tagger->label(s.cls);
                            sound_events.push_back({label ? label : "", s.start, s.end, s.peak});
                        }
                        have_sounds = true;
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "parakeet-server: sound tagging error: %s\n", e.what());
                    }
                }
            }
            Response out = format_transcription(tr, fmt, duration_sec, include_words,
                                                have_sounds ? &sound_events : nullptr);
            res.set_content(out.body, out.content_type.c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "parakeet-server: inference error: %s\n", e.what());
            res.status = 500;
            res.set_content(error_body("internal transcription error", "server_error"),
                            "application/json");
        }
    });

    std::fprintf(stderr, "parakeet-server: listening on http://%s:%d (model: %s)\n",
                 host.c_str(), port, model_path.c_str());
    bool ok = svr.listen(host.c_str(), port);
    g_server = nullptr;
    {
        std::lock_guard<std::mutex> lock(models_mu);
        stopping = true;
    }
    models_cv.notify_all();
    if (unloader.joinable()) unloader.join();
    models.clear();
    pk::shutdown_backend();
    if (!ok) {
        std::fprintf(stderr, "parakeet-server: failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
