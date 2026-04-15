# Evie Voice Loop: Sentence-Level Streaming TTS + Filler Phrases

## Context

Currently the voice loop generates the **entire LLM response** before passing it to TTS. For long responses this creates a noticeable silence gap. Two improvements:

1. **Sentence-level streaming**: Start TTS on the first sentence while the LLM generates the rest
2. **Filler phrases**: Play natural fillers ("Hmm...", "That's a good question...") while the LLM prefills

## Current Architecture (sequential)

```
[silence] → LLM generates full response → kokoro.create_stream(full_text) → play audio
```

- `llm_generate()` calls `generate()` which wraps `stream_generate()` internally but collects all tokens before returning
- `play_tts_stream(response)` receives the complete text, passes to `kokoro.create_stream(text)`
- Kokoro splits text into phoneme batches internally and streams audio chunks via async generator

## Proposed Architecture (pipelined)

```
[filler TTS plays] → LLM streams tokens → buffer until sentence end →
  sentence 1 → kokoro TTS → play audio (while LLM continues generating)
  sentence 2 → kokoro TTS → play audio (while next sentence generates)
  ...
```

## Implementation Plan

### Part A: Sentence-Level Streaming

#### Step A1: Replace `llm_generate()` with `llm_stream_generate()`

New function that yields text chunk-by-chunk using `stream_generate` from mlx_vlm (already available at `.venv/.../mlx_vlm/generate.py:575`):

```python
from mlx_vlm import stream_generate

def llm_stream_generate(messages, max_tokens=200, temperature=0.7, **kwargs):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    for response in stream_generate(model, processor, prompt,
                                     max_tokens=max_tokens, temperature=temperature,
                                     repetition_penalty=1.2, **kwargs):
        yield response.text
```

Each yield produces a small text fragment (typically 1 token decoded to text).

#### Step A2: Add sentence buffer

Accumulate tokens into a buffer. When a sentence boundary is detected (`. `, `! `, `? `, or end of generation), yield the complete sentence:

```python
def llm_stream_sentences(messages, **kwargs):
    buffer = ''
    for chunk in llm_stream_generate(messages, **kwargs):
        buffer += chunk
        while True:
            match = re.search(r'[.!?](?:\s|$)', buffer)
            if match:
                sentence = buffer[:match.end()].strip()
                buffer = buffer[match.end():]
                if sentence:
                    yield sentence
            else:
                break
    if buffer.strip():
        yield buffer.strip()
```

#### Step A3: Rewrite `play_tts_stream()` to accept a sentence generator

Instead of taking a complete response string, take a generator of sentences and pipeline them:

```python
def play_tts_streamed(sentence_gen):
    for sentence in sentence_gen:
        tts_stream = kokoro.create_stream(sentence, voice=args.voice, ...)
        asyncio.run(_play_sentence(tts_stream))
        if interrupted:
            break
    return full_response_text
```

Key design: each sentence gets its own `kokoro.create_stream()` call. The pipeline becomes:

```
LLM generates sentence N+1 → kokoro synthesises sentence N → speaker plays sentence N-1
```

Three stages running concurrently.

#### Step A4: Thread the pipeline

Since `stream_generate` is synchronous (yields from MLX) and `kokoro.create_stream` is async:
1. Run LLM streaming in the main thread, buffering sentences into a `queue.Queue`
2. Run a TTS worker thread that pulls sentences from the queue, synthesises, and pushes audio to a playback queue
3. Playback happens via sounddevice OutputStream (already async callback-based)

MLX operations are NOT thread-safe — LLM generation must stay on the main thread.

#### Step A5: Modify `process_utterance()`

Replace the current sequential flow:
```python
response = llm_generate(messages)
play_tts_stream(response)
```

With:
```python
sentence_gen = llm_stream_sentences(messages)
response = play_tts_streamed(sentence_gen)
```

The full response text is still collected for history storage.

### Part B: Filler Phrases

#### Step B1: Define filler phrases

```python
FILLERS = [
    "Hmm...",
    "That's a good question...",
    "Let me think...",
    "Sure...",
    "Right...",
    "Okay...",
    "Well...",
]
```

#### Step B2: Play filler while LLM prefills

The LLM's biggest latency is the **prefill step** (processing the full prompt before generating the first token). This is where the silence gap lives.

After transcription, before LLM generation:
1. Pick a random filler
2. Start TTS synthesis of the filler in a background thread
3. Start LLM generation (prefill + first tokens)
4. Play filler audio while prefill runs
5. When first LLM sentence is ready, transition to the real streamed response

```python
import random

def play_filler_and_generate(messages, **kwargs):
    filler = random.choice(FILLERS)
    filler_audio = kokoro.create(filler, voice=args.voice, ...)
    sentence_gen = llm_stream_sentences(messages, **kwargs)
    play_audio(filler_audio)
    response = play_tts_streamed(sentence_gen)
    return filler + ' ' + response
```

#### Step B3: Smart filler selection

Not every utterance needs a filler. Use heuristics:
- **Questions** (transcript ends with `?` or contains "what", "how", "why"): Thinking fillers ("That's a good question...", "Let me think...")
- **Commands** ("do this", "run that"): Acknowledgment fillers ("Sure...", "Okay...", "Right...")
- **Short/simple utterances**: Skip fillers entirely (latency already low)
- **Follow-up turns** (history > 0): Less likely to need fillers (context already primed, prefill faster)

#### Step B4: Add `--filler` CLI flag

```python
p.add_argument('--filler', '--no-filler', action=argparse.BooleanOptionalAction, default=True)
```

Default ON since it improves perceived responsiveness.

## Latency Analysis

| Phase | Current | With streaming | Improvement |
|-------|---------|---------------|-------------|
| LLM prefill | ~500ms silence | Filler plays during | Perceived: **0ms** |
| First sentence to audio | ~2-4s (full gen + TTS) | ~800ms (first sentence gen + TTS) | **~3x faster** |
| Full response complete | Same | Same | No change |

## Key Files

- `evie-mac.py` — all changes (new streaming functions, modified process_utterance)
- No new config files needed
- `mlx_vlm.stream_generate` — already available, just need to import
- `kokoro.create_stream` — already used, no changes needed

## Risks & Mitigations

1. **Sentence boundary detection**: LLM may not always use clear sentence endings. Mitigation: also split on `,` after a minimum length (80+ chars), and always flush on generation end.
2. **TTS quality at sentence boundaries**: Each sentence is a separate kokoro call, so prosody won't flow across. Mitigation: Kokoro handles standalone sentences well. Buffer very short fragments until minimum length.
3. **Barge-in between sentences**: Current AEC interrupt detection happens within `_play()`. With sentence-level streaming, also check between sentences — natural interruption point.
4. **Thread safety**: MLX is not thread-safe. LLM generation stays on main thread. TTS and playback in worker threads.

## Verification

1. Time first-audio-output with and without streaming on the same prompt
2. Listen for natural prosody at sentence boundaries
3. Test barge-in works mid-response
4. Test filler phrases don't overlap with LLM response
5. Test long responses (5+ sentences) stream smoothly
6. Test short responses (1 sentence) don't add unnecessary filler delay
