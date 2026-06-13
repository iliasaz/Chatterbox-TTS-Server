# Generation Parameters per Model

This server can serve three Chatterbox model families. They share most of the API
surface but accept different generation parameters: a value sent to the wrong model
is either silently ignored (with a warning) or, in one case, raises a `TypeError`.
Use this page to know exactly which knobs do something on which model.

## Models

| Selector (`model.repo_id`) | Class | What it's for |
|---|---|---|
| `chatterbox-turbo` | `ChatterboxTurboTTS` | Fast English model. The **only** model with native paralinguistic tags (see `paralinguistic-tags.md`). |
| `chatterbox` | `ChatterboxTTS` | Original English model. Slower than Turbo, but accepts the expressiveness/pacing knobs (`exaggeration`, `cfg_weight`). |
| `chatterbox-multilingual` | `ChatterboxMultilingualTTS` | 23-language model. Selectable T3 weights (`v2` or `v3` via `model.multilingual_t3_version`). Current durable default in `config.yaml`. |

Switch models from the UI (Active Model dropdown → Apply & Restart) or via
`/save_settings` + `/restart_server`. The active model on startup is whatever
`config.yaml` says.

## Parameter support matrix

| Parameter | `chatterbox-turbo` | `chatterbox` (original) | `chatterbox-multilingual` |
|---|---|---|---|
| `temperature` | yes | yes | yes |
| `top_p` | yes | yes | yes |
| `top_k` | **yes** | **not accepted (raises TypeError if passed directly)** | **not accepted** |
| `repetition_penalty` | yes | yes | yes |
| `min_p` | accepted, **ignored** (warned) | yes | yes |
| `exaggeration` | accepted, **ignored** (warned) | yes | yes |
| `cfg_weight` | accepted, **ignored** (warned) | yes | yes |
| `seed` | yes | yes | yes |
| `language` | en only | en only | yes (23 languages) |
| `multilingual_t3_version` | n/a | n/a | yes (`v2` or `v3`) |
| Paralinguistic tags in `text` | **yes** (19 tokens) | no (text spoken/ignored) | no |
| Voice reference / predefined voice | yes | yes | yes |
| `speed_factor` | **accepted but ignored** | accepted but ignored | accepted but ignored |

Notes on the matrix:

- The server's `engine.synthesize()` builds a shared kwargs dict for all models and
  only adds `top_k` when the active model is Turbo, so callers can safely pass
  `top_k` on any model from the API — it just won't reach the original/multilingual
  paths.
- The OpenAI-compatible endpoint (`/v1/audio/speech`) uses the server's
  generation defaults for `temperature`/`exaggeration`/`cfg_weight`/`top_p`/`top_k`/
  `repetition_penalty`; only `seed` and `language` come from the request.

## What each parameter does to the sound

These parameters sample over **discrete acoustic tokens** (not text), so
"randomness" here means variation in prosody, timing, and pronunciation — not
word choice. Default values come from `config.yaml → generation_defaults`.

### `temperature` (default 0.8, range 0.0–1.5)

Master expressiveness ↔ stability dial.

- Lower (~0.5): flatter, more monotone, very consistent. Tags/emotion may be
  under-realized. Russian `ё` and other weak spots become more stable.
- Higher (~1.0+): livelier, more varied intonation, but more risk of slurring,
  odd pitch, mispronunciation, glitches.

The single most useful knob across all three models. The only model knob you
get on Turbo — the others (`exaggeration`/`cfg_weight`) are ignored there.

### `top_p` (default 0.95, range 0.0–1.0)

Nucleus sampling — keep the smallest token set whose cumulative probability ≥ `p`.

- Lower (~0.8): tighter, safer, more stable delivery, less expressive.
- Higher (toward 1.0): fuller distribution → more variation and expressive
  detail, more chance of off-distribution warbles.

### `top_k` (default 1000, range 0–2000) — Turbo only

Keep only the `k` most-likely acoustic tokens each step. The codebook is ~6561
usable tokens, so 1000 is permissive.

- Lower (~50–100): cleaner, safer, more conservative; less variety.
- Very low (~5–10): can sound robotic/flat.

Original and multilingual models do not accept `top_k` — the server drops it
for those.

### `repetition_penalty` (default 1.2, range 1.0–2.0)

Penalizes already-emitted acoustic tokens across the whole sequence. Acts as the
**anti-stutter / anti-loop** lever in acoustic-token space.

- `1.0`: disables it — risk of audible repetition or droning.
- ~`1.2`: sweet spot for both stability and naturalness.
- `>1.5`: starts to **backfire** — penalizes legitimately recurring frames
  (sustained vowels, silence, steady pitch), causing unnatural pitch drift,
  rushed/clipped delivery, or skipped content.

### `exaggeration` (default 0.5) — original & multilingual only

Controls expressiveness/animation. Higher → more dramatic delivery; lower → more
neutral. **Turbo ignores this** (and warns). On Turbo, equivalent control comes
from the paralinguistic emotion tags (`[angry]`, `[happy]`, `[dramatic]`, etc.).

### `cfg_weight` (default 0.5) — original & multilingual only

Classifier-Free Guidance weight — influences how strictly the model follows the
prompt/reference style and affects pacing.

- Lower: looser adherence, more variation.
- Higher: tighter adherence to the reference voice/style.

**Turbo ignores this** as well.

### `min_p` (default 0.05) — original & multilingual only

Minimum-probability filtering. Turbo ignores it; it's not exposed in the server
UI/API by default (uses each model's library default).

### `seed` (default 0 → random)

`0` means random. Any non-zero integer makes the output reproducible (useful
when you want to lock a good take, or A/B params). The server applies it globally
before generation.

### `speed_factor` — **deprecated**

Once applied a post-hoc time-stretch to the audio. It introduced artifacts, so
the server no longer applies it on any path. The field is still accepted on
`/tts` (`speed_factor`) and the OpenAI endpoint (`speed`) for backward
compatibility, but has no effect.

If you actually need a different playback speed, do it client-side after
generation rather than asking the engine for it.

## Multilingual specifics

### Language

`language` (alias `language_id`) accepts: `ar, da, de, el, en, es, fi, fr, he, hi,
it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh` (23 total). It's a runtime
parameter on `/tts` requests and on the OpenAI endpoint.

`language` is only meaningful for `chatterbox-multilingual`. On the English-only
models the value is accepted but the model only synthesizes English.

### T3 weights version (`v2` vs `v3`)

`model.multilingual_t3_version` selects which T3 checkpoint the multilingual
model loads:

- `v3` (default, **current durable default**) — the newer T3 weights added in the
  resemble-ai/chatterbox v3 commit. Better overall.
- `v2` — the previous T3 weights, kept as a fallback.

Changeable from the UI (dropdown next to the Active Model selector when
multilingual is selected) or via `/save_settings` + restart. Both checkpoints
are pre-cached in the `hf_cache` Docker volume, so switching is fast.

### Russian text preprocessing (auto-applied, multilingual only)

When `language='ru'`, the server's `add_russian_stress` monkey-patch runs
inside the tokenizer's `encode()`, between its `preprocess_text` (lowercase +
NFKD) and the final HF tokenizer call. The pipeline is four steps:

1. **NFC recompose** — required because ruaccent's internal `normalize` regex
   has an allow-list that does not include combining marks (`U+0308`,
   `U+0306`), so it silently strips them from NFKD-decomposed input,
   destroying `ё` and `й` before stressing. NFC restores them.
2. **`ruaccent.process_all`** — preserves `ё`/`й`, and marks stress as `+`
   before the stressed vowel (skips `ё` since it's inherently stressed).
3. **`+` → `U+0301`** — convert ruaccent's stress format to combining acute
   *after* the vowel, which is what the model expects.
4. **NFKD decompose** — back to `е + U+0308` for `ё` and `и + U+0306` for
   `й`. The model's composed `ё`/`й` tokens (2374/2373) rendered as silent
   in listening tests; the decomposed form is what the model actually handles
   (it sounds like `/e/`-ish for `ё`, which is wrong vowel but predictable and
   pronounced — better than silence).

You can pre-mark stress manually by inserting `U+0301` after a vowel
(e.g. `приве́т`); those marks pass through unchanged.

The ruaccent model (~700 MB, turbo3.1 + dictionary) downloads to the
`hf_cache` volume on first Russian synthesis and is reused thereafter.

**Residual `ё` quality** is a model-level limitation: with this pipeline,
`ё` words pronounce as `/e/`-ish (close to `тёплый → теплый`) — recognizable
but not the correct `/o/`-after-soft-consonant sound. There is no known
encoding that gives correct `ё` reliably on this model; the composed `ё`
token (2374) drops entirely, the decomposed form gives `/e/`. Use lower
`temperature` / a fixed `seed` for more stable runs.
