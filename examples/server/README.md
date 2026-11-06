# parakeet-server

A small OpenAI-drop-in HTTP server for transcription, built on parakeet.cpp.
Point any OpenAI client's `base_url` at it and call
`POST /v1/audio/transcriptions`.

This is an example, not a production service. It loads models on demand from
the `--model` directory, runs one transcription at a time by default, and
accepts WAV uploads only.

For a production deployment, use [LocalAI](https://localai.io), which embeds
parakeet.cpp as a backend and adds the things this example deliberately leaves
out: a model gallery, the full OpenAI API
surface, auth, and metrics.

## Build

Built by default with the rest of the project (`PARAKEET_BUILD_SERVER=ON`):

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

A request may name another GGUF from the same directory (see `model` under
Supported). A model that served no request for `--unload-after` seconds is
unloaded (default 300, `0` keeps models loaded) and loaded again by the next
request that names it.

With a published model by alias, downloaded once into the model directory,
`${XDG_CACHE_HOME:-$HOME/.cache}/parakeet.cpp/models` unless set with
`--model-dir` or `PARAKEET_MODEL_DIR`. `--fetch` downloads models into that
directory without serving:

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
    # empty model uses the server default; the OpenAI-compatible API requires the model param
    print(client.audio.transcriptions.create(model="", file=f).text)
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

- `model`: optionally override the model for this request with the file name of
  another GGUF, for example `tdt-0.6b-v3-q8_0.gguf`. Empty uses the server
  default.
- `response_format`: `json` (default), `text`, `verbose_json`.
- `boost`: words or phrases to favor during TDT decoding, one per line, each
  followed by its non-negative strength, for example `symlink 4`, `home dir 4`,
  `code 8`. The field may be repeated. Matching ignores English letter case and
  follows word boundaries; punctuation in the output is preserved. The bonus is
  distributed across a phrase's characters, with partial-match bonuses cancelled
  on a mismatch; matches ending at the same boundary use the strongest completed
  bonus. Bonuses are applied before token normalization, so this is contextual
  logit biasing, not an external language model. `0` leaves scores unchanged.
  Boosted requests use greedy decoding and support `json` or `text`.
- `spelling`: output spelling followed by a tab and one alternative, one mapping
  per line. For example, `Ghostty<TAB>ghost ty` replaces `ghost ty` with `Ghostty`
  (`<TAB>` means a literal tab). The field may be repeated. Matching ignores ASCII
  case and `Ä/ä`, `Ö/ö`, `Ü/ü`, and requires whole words or phrases. Other letters,
  spaces and dashes match literally. Only listed alternatives match. The longest
  match wins; ties use the first mapping. Replacements are not matched again.
  Corrections affect transcript text in all response formats, including segment
  text in `verbose_json`; timestamped `words` entries remain as recognized.
- `timestamp_granularities[]=word` adds a `words` array to `verbose_json`.
- `--sound-model` adds a `sound_events` array to `verbose_json`.

## Known simplifications

- WAV uploads only. Other formats return 400. Convert with ffmpeg first.
- `verbose_json` emits a single `segment` spanning the whole transcript;
  Parakeet has no native segmentation. Word timestamps are real.
- `language` in `verbose_json` is a fixed `en` placeholder.
- `temperature` and `prompt` are accepted and ignored (greedy decode).
- Inference is serialized by a mutex. For real parallelism, hold a pool of
  `pk::Model` contexts instead of one.
