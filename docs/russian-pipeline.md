# Russian Text Preprocessing Pipeline

Applies only when `language='ru'` on the **multilingual** model. Implemented as
a monkey-patch of `chatterbox.models.tokenizers.tokenizer.add_russian_stress`
in `engine.py`. The patch runs inside the tokenizer's `encode()`, between
`preprocess_text` (lowercase + NFKD) and the final HF tokenizer call.

## The pipeline

```
preprocess_text (lowercase + NFKD)        ← upstream, in the tokenizer
        ↓
[1]  NFC recompose                        ← restores combining marks for ruaccent
        ↓
[2]  ruaccent.process_all                 ← preserves ё/й, marks stress as '+' before vowel
        ↓
[3]  '+'  →  U+0301                       ← convert stress to combining acute after vowel
        ↓
[4]  NFKD decompose                       ← ё → е + U+0308,  й → и + U+0306
        ↓
prepend "[ru]" + space replacement        ← upstream, in the tokenizer
        ↓
HF tokenizer.encode → model
```

## Why each step

### [1] NFC recompose

`ruaccent`'s internal `normalize` regex has an allow-list that does **not**
include combining marks `U+0308` (diaeresis) or `U+0306` (breve). When fed
NFKD-decomposed Russian, that regex strips the marks before stressing — so
`ё → е` and `й → и` *silently inside ruaccent*. That was the cause of
`тёплый → теплый`, `жёлтый → желтый`, `мой → мои`.

NFC recomposes those marks back onto their base letters (`е + ◌̈ → ё`,
`и + ◌̆ → й`) so ruaccent sees real Russian and preserves them.

### [2] ruaccent.process_all

ruaccent **preserves** `ё`/`й` (now that it sees them composed) and inserts a
`+` immediately before the stressed vowel of every other word. It does not
mark `ё` words because `ё` carries inherent stress in its lexicon.

Examples (composed input → ruaccent output):

| input | ruaccent |
|---|---|
| `тёплый` | `тёплый` |
| `ёжик` | `ёжик` |
| `жёлтый` | `жёлтый` |
| `мой` | `м+ой` |
| `йога` | `й+ога` |
| `покойный` | `пок+ойный` |
| `русский` | `р+усский` |

### [3] `+` → `U+0301`

The model expects the stress as a combining acute **after** the stressed
vowel. ruaccent emits it as `+` **before** the vowel. We rewrite each
`+<vowel>` to `<vowel>́`. So `м+ой` becomes `мо́й`.

### [4] NFKD decompose

The model's composed `ё` (token 2374) and `й` (token 2373) rendered as
**silent / dropped** in listening tests (e.g., `ёжик → жик`,
`йога → ога`). Decomposing them back to `е + U+0308` and `и + U+0306`
gives the model a representation it actually pronounces — `ё` comes out
`/e/`-ish (wrong vowel but predictable and audible), `й` is restored to
its NFKD form. The acute stress marks added in step 3 already attach to
their base vowels and survive NFKD unchanged.

## Worked example: `"Тёплый ёжик"`

(`◌̈` = U+0308 diaeresis, `◌̆` = U+0306 breve, `◌́` = U+0301 acute.)

| Step | Result |
|---|---|
| Client sends | `"Тёплый ёжик"` |
| `preprocess_text` (lowercase + NFKD) | `"т е◌̈ п л ы и◌̆   е◌̈ ж и к"` |
| Hook [1] NFC recompose | `"тёплый ёжик"` |
| Hook [2] ruaccent | `"тёплый ёжик"` (no `+` — `ё` inherent) |
| Hook [3] `+` → `U+0301` | `"тёплый ёжик"` (nothing to convert) |
| Hook [4] NFKD decompose | `"т е◌̈ п л ы и◌̆   е◌̈ ж и к"` |
| Tokenizer prepends `[ru]` + tokenizes | `[<ru>, т, е, ◌̈, п, л, ы, и, ◌̆, SPACE, е, ◌̈, ж, и, к]` |

## A non-`ё` example: `"мой"`

| Step | Result |
|---|---|
| `preprocess_text` | `"м о и◌̆"` |
| Hook [1] NFC | `"мой"` |
| Hook [2] ruaccent | `"м+ой"` |
| Hook [3] `+` → `U+0301` | `"мо́й"` |
| Hook [4] NFKD decompose | `"м о ◌́ и◌̆"` |
| Tokens | `[м, о, ◌́, и, ◌̆]` |

The stress (`◌́`) ends up attached to `о` (correct), and `й` is in the
NFKD form the model handles.

## What this fixes vs. what it doesn't

**Fixed:**
- Stress is now placed correctly on every word in the sentence, including
  the words next to `ё`/`й`. Before this pipeline, ruaccent's regex was
  silently stripping NFKD marks before processing, so stress fell on the
  wrong letters and `ё`/`й` words came out vowel-stripped (`теплый`, `мои`).

**Known model limitation (not preprocessing):**
- `ё` words come out as `/e/`-ish rather than the correct
  palatalized-consonant-plus-`/o/` sound (`тёплый` sounds close to
  `теплый`). There is no encoding we've found that yields the correct
  `ё` reliably on this model — the composed `ё` token drops, the
  decomposed form sounds `/e/`. Treat this as a model-level limit.

**Per-request levers** for marginal cases:

- Lower `temperature` (e.g. 0.65) and a fixed non-zero `seed` for
  reproducible, less variable runs.
- Insert `U+0301` manually after a vowel in input text to override
  stress for a specific word; those marks pass through the pipeline
  unchanged.
- Try a different predefined voice — `ё`/`й` quality varies by voice.

## Manual override examples

```
прив+ет          ← won't work; '+' is only ruaccent's intermediate format
приве́т           ← works; U+0301 after е is what the model wants
бельё            ← will go through pipeline; ё decomposed; model says /e/
```

## Source

`engine.py`, in the `_ruaccent_add_russian_stress` function. The monkey-patch
target is `chatterbox.models.tokenizers.tokenizer.add_russian_stress`. ruaccent
itself is installed via `requirements.txt`; its ~700 MB model
(`omograph_model_size="turbo3.1"`, `use_dictionary=True`) lazy-loads into the
`hf_cache` Docker volume on first Russian synthesis.
