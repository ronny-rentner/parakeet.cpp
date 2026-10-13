# parakeet-server

A small OpenAI-drop-in HTTP server for transcription, built on parakeet.cpp.
Point any OpenAI client's `base_url` at it and call
`POST /v1/audio/transcriptions`.

This is an example, not a production service. It keeps requested models loaded,
runs one transcription at a time by default, and accepts WAV uploads only.

For a production deployment, use [LocalAI](https://localai.io), which embeds
parakeet.cpp as a backend and adds the things this example deliberately leaves
out: a model gallery, the full OpenAI API
surface, auth, and metrics.

## Build

Built by default with the rest of the project (`PARAKEET_BUILD_SERVER=ON`):

The experimental phrase-boosting API requires `nlohmann-json3-dev` on Debian.

```sh
cmake -B build && cmake --build build --target parakeet-server -j
```

## Run

With a local model:

```sh
./build/examples/server/parakeet-server --model path/to/model.gguf --port 8080
```

To run several transcriptions at once, add `--concurrency K` (default 1). The
server then keeps K CPU backends, and `--threads` is the thread count of each
one. Keep `K x threads` at or below the physical cores. See
[`docs/concurrency.md`](../../docs/concurrency.md).

```sh
./build/examples/server/parakeet-server --model path/to/model.gguf --concurrency 4 --threads 2
```

With a published model by alias (downloaded once and cached under
`${XDG_CACHE_HOME:-$HOME/.cache}/parakeet.cpp/models`, override with
`--cache-dir` or `PARAKEET_CACHE_DIR`):

```sh
./build/examples/server/parakeet-server --model tdt_ctc-110m-q4_k
```

`--model` accepts a local `.gguf` path, an `http(s)://` URL, a `<name>.gguf`
filename in the `mudler/parakeet-cpp-gguf` repo, or one of these aliases:

| Alias                | Model                                   |
|----------------------|-----------------------------------------|
| `tdt_ctc-110m`       | hybrid TDT+CTC 110M (f16)               |
| `tdt_ctc-110m-q4_k`  | hybrid TDT+CTC 110M (q4_k, smallest)    |
| `tdt_ctc-1.1b`       | hybrid TDT+CTC 1.1B (f16)               |
| `tdt-0.6b-v2`        | TDT 0.6B v2 (f16)                       |
| `tdt-0.6b-v3`        | TDT 0.6B v3, multilingual (f16)         |
| `tdt-1.1b`           | TDT 1.1B (f16)                          |
| `ctc-0.6b`           | CTC 0.6B (f16)                          |
| `ctc-1.1b`           | CTC 1.1B (f16)                          |
| `rnnt-0.6b`          | RNN-T 0.6B (f16)                        |
| `rnnt-1.1b`          | RNN-T 1.1B (f16)                        |
| `eou-120m`           | realtime EOU 120M (f16)                 |

Downloads use `curl` (or `wget`). If neither is on `PATH`, download the `.gguf`
yourself and pass the local path.

## Docker

A prebuilt image is published per push to `ghcr.io/<owner>/parakeet.cpp-server`
(CPU by default, `:latest-cuda` for the CUDA build). It binds `0.0.0.0` and
exposes port 8080. Pass the same `--model` argument you would on the CLI;
mount a local `.gguf`, or let it fetch an alias on first run:

```sh
# serve a published model by alias (downloaded into the container)
docker run --rm -p 8080:8080 ghcr.io/mudler/parakeet.cpp-server --model tdt_ctc-110m

# serve a local model (mount it read-only)
docker run --rm -p 8080:8080 -v "$PWD/model.gguf:/model.gguf:ro" \
  ghcr.io/mudler/parakeet.cpp-server --model /model.gguf
```

## Call it

```sh
curl -F file=@audio.wav -F response_format=verbose_json \
  http://localhost:8080/v1/audio/transcriptions
```

With the OpenAI Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="parakeet", file=f).text)
```

## Sound events

Pass `--sound-model <ced.gguf>` with a local path to a ced.cpp sound-event
tagger GGUF. When set, a `verbose_json` response gains a `sound_events` array,
one entry per detected event: `{"label","start","end","score"}`. `json` and
`text` responses are unchanged, and the sound pass only runs for
`verbose_json` requests, so other requests never pay for it. `--sound-model`
takes a local path only, it is not resolved through the alias/URL model
fetcher used by `--model`.

## Supported

- `model`: a path, URL, or alias, using the same resolver as `--model`. The first
  request loads it; later requests reuse it. Models remain loaded until the
  server exits. Omitted, empty, or `parakeet` selects the startup model.
  Loading is serialized; an already-running transcription can continue during
  loading, but new requests wait for the load to finish.
- `response_format`: `json` (default), `text`, `verbose_json`.
- `boost_phrase`: repeat this field for each word or phrase to favor during TDT
  decoding. Matching ignores English letter case and follows word boundaries;
  punctuation in the output is preserved. Uses greedy decoding.
- `boost_score`: non-negative strength, default `2` when phrases are supplied;
  `0` leaves scores unchanged. The bonus is distributed across a phrase's
  characters, with partial-match bonuses cancelled on a mismatch. Bonuses are
  applied before token normalization, so this is contextual logit biasing,
  not an external language model. Boosted requests support `json` or `text`.
- `timestamp_granularities[]=word` adds a `words` array to `verbose_json`.
- `boost_phrases`: JSON object mapping each word or phrase to its own non-negative
  strength, for example `{"symlink":6,"home dir":4,"grammar":3}`. Use this field
  instead of `boost_phrase` and `boost_score`; combining them returns an error.
  Matching and score application follow the same rules as shared-strength boosting.
  Matches ending at the same boundary use the strongest completed bonus.
- `--sound-model` adds a `sound_events` array to `verbose_json`.

## Known simplifications

- WAV uploads only. Other formats return 400. Convert with ffmpeg first.
- `verbose_json` emits a single `segment` spanning the whole transcript;
  Parakeet has no native segmentation. Word timestamps are real.
- `language` in `verbose_json` is a fixed `en` placeholder.
- `temperature` and `prompt` are accepted and ignored (greedy decode).
- Inference is serialized by a mutex. For real parallelism, hold a pool of
  `pk::Model` contexts instead of one.
