# Paralinguistic Tags (Turbo only)

Chatterbox-Turbo recognizes **19 dedicated bracket tags** that steer non-speech
sounds, emotion, and delivery style. These are real **single tokens** in the
model's tokenizer (`added_tokens.json`, IDs 50257–50275) — not parsed text — so
the bracket convention is a **fixed registry**: only the exact 19 strings work.
Arbitrary words in brackets (e.g. `[whisper]`, `[smile]`, `[mumble]`) get
BPE-split into letters and are read out / mangled.

**These tags only have an effect on `chatterbox-turbo`.** On the original or
multilingual model, the same brackets either get spoken literally or are
ignored.

## The full registry

Three behavioral categories. Use the right category in the right position.

### 1. Sound events (9) — inline reactions

| Tag | What it does |
|---|---|
| `[laugh]` | full laugh |
| `[chuckle]` | small, restrained laugh |
| `[sigh]` | sigh |
| `[gasp]` | sharp intake of breath |
| `[cough]` | cough |
| `[clear throat]` | throat clear |
| `[sniff]` | sniff |
| `[groan]` | groan |
| `[shush]` | "shh"/shushing sound |

**Behavior:** these are like vocal punctuation — they fire at the spot you put
them. Use them **inline**, mid-sentence, wherever a reaction would land.

### 2. Emotions (6) — delivery directives

| Tag | What it does |
|---|---|
| `[angry]` | angry delivery |
| `[fear]` | scared delivery |
| `[surprised]` | surprised |
| `[crying]` | crying / tearful |
| `[happy]` | happy / upbeat |
| `[sarcastic]` | sarcastic |

**Behavior:** these set the **mode for the speech that follows**, like a mood
switch. They are NOT inline events. Use one at the **start** of a sentence (or
the segment you want them to color).

### 3. Styles (4) — delivery directives

| Tag | What it does |
|---|---|
| `[whispering]` | whispered delivery |
| `[dramatic]` | dramatic, slow, weighty |
| `[narration]` | narrator / audiobook voice |
| `[advertisement]` | ad-read / energetic |

**Behavior:** same as Emotions — prefix a sentence to set the style for it.

## How to mix them — and how not to

### Sound events: mix freely

`[laugh]`, `[sigh]`, etc. are point events, so combining multiple within one
sentence works as you'd expect:

```
Oh, [chuckle] that's hilarious! [cough] Excuse me.
```

You can put a sound event inside an emotion- or style-styled sentence — the
sound fires, the emotion/style still colors the rest:

```
[happy] I can't believe it! [laugh] This is wonderful.
```

### Emotions / styles: one per generation

This is the most important rule and the easiest to get wrong.

**One generation = one utterance = one emotion/style.** The server generates
**per chunk** (each chunk is an independent autoregressive run), and a single
generation only carries one mood — the dominant tag wins, the others are
effectively silent.

So this looks like style switching, but it isn't:

```
[crying] The house was silent. [whispering] Then the door slammed.
```

If those two sentences land in **one** chunk, you get one delivery (probably
just one of the two emotions) across both. To make them sound different, force
each sentence into its own chunk: keep "Split text into chunks" on, and set the
chunk size near (or below) the length of a single sentence — verified to split
the line above when `chunk_size=50`. Equivalent via API:

```json
{"text": "[crying] The house was silent. [whispering] Then the door slammed.",
 "split_text": true, "chunk_size": 50}
```

Or send the styled segments as separate `/tts` requests.

### Don't combine two emotion/style tags on the same sentence

```
[happy] [dramatic] I won the prize!     <- bad: one wins, the other is lost
```

Pick one. If you genuinely want a blend, that's not how this model works.

### Do combine a sound event with an emotion/style on the same sentence

```
[dramatic] And then... [gasp] he turned around.     <- fine
```

The emotion/style tag colors the sentence; the sound fires at its position.

### Rule of thumb

| Want | Where to put the tag |
|---|---|
| A vocal reaction at a moment | Sound event, inline at that moment |
| A whole sentence delivered with feeling | Emotion or style, at the start of the sentence |
| Switch feeling partway through | Split into separate sentences/chunks, one tag each |
| Tone for a long passage | Single tag at the start of the passage |

## Format requirements

Tags are matched as exact tokens, so format is strict:

- **Brackets touching the word**: `[whispering]` good; `[ whispering ]` bad
  (becomes 3 BPE pieces — `[`, `whispering`, `]` — and may be read out as letters).
- **Lowercase only**: `[whispering]` good; `[Whispering]` bad.
- **Exact spelling**: `[whispering]` good; `[whisper]` bad (not in the registry).
- **Single internal space** where used: `[clear throat]` good; `[clearthroat]` or
  `[clear  throat]` bad.

The UI tag buttons always insert the canonical form — if you hand-type tags,
double-check against the table above.

## Other ways to instruct the model

The 19 tags aren't the only way to shape delivery; complement them with:

- **Filler words in the text** ("um", "uh") — the model renders them naturally
  as disfluency, helpful for casual/conversational delivery.
- **Punctuation as pacing** — periods, commas, ellipses, dashes all affect
  intonation and timing. The text-norm step keeps `. ! ? - ,` and collapses
  `…` and `:` into commas.
- **Reference voice / predefined voice** — carries a lot of style on its own.
  Picking a different voice often does more than any tag.
- **Sampling parameters** — see `parameters.md`. Lower `temperature` makes the
  delivery (including tags) more conservative; higher amplifies it.

## What about original and multilingual?

Neither model has a paralinguistic tag system. On those models:

- The original supports `exaggeration` and `cfg_weight` for delivery shaping
  (see `parameters.md`); use those instead of tags.
- The multilingual exposes the same `exaggeration`/`cfg_weight` plus 23
  languages.
- Bracketed text on these models is either spoken literally or ignored; don't
  expect it to do anything.

If you want both tag-driven delivery **and** Russian (or any non-English
language), you can't have both in one generation — pick the model that matches
the language you need.
