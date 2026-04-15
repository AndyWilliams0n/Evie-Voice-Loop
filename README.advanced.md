# Evie — Advanced Usage & Technical Details

For basic setup, see [README.md](README.md).

## All Command-Line Options

```bash
// Recommended defaults (TTS + smart turn + voice interrupt all on)
uv run evie-mac.py

// Enable chimes and soft ticks while generating
uv run evie-mac.py --chime

// Persistent memory (reads/writes MEMORY.md)
uv run evie-mac.py --memory

// Text-only mode (no TTS)
uv run evie-mac.py --no-tts

// Disable voice interruption (keypress only)
uv run evie-mac.py --no-vpio

// Disable smart turn detection
uv run evie-mac.py --no-smart-turn

// Different voice
uv run evie-mac.py --voice bf_emma

// Use the smaller E2B model (faster, slightly lower quality)
uv run evie-mac.py --model mlx-community/gemma-4-E2B-it-4bit

// Custom silence timeout (ms before Evie considers you done speaking)
uv run evie-mac.py --silence-ms 500

// Debug: record mic stream to a WAV file
uv run evie-mac.py --record
```

## Kokoro Voice Options

| Voice | Accent | Gender | Quality | Notes |
|-------|--------|--------|---------|-------|
| `af_heart` | US | Female | Grade A | Default |
| `af_bella` | US | Female | Grade A- | HH training |
| `bf_emma` | UK | Female | Grade B- | HH training |
| `am_fenrir` | US | Male | Grade C+ | H training |
| `am_puck` | US | Male | Grade C+ | H training |
| `am_michael` | US | Male | Grade C+ | H training |
| `bm_fable` | UK | Male | Grade C | MM training |
| `bm_george` | UK | Male | Grade C | MM training |

## Architecture

```
Mic (16kHz) --> Silero VAD --> Smart Turn --> Moonshine --> Gemma 4 E4B --> Kokoro --> Speakers
                                                                ^                       |
                                                    SOUL.md + MEMORY.md                 |
                                                                                        v
Mic during TTS --> macOS VPIO (AEC + noise suppression) --> Silero VAD --> voice interrupt <--+
```

## How It Works

1. **Mic capture** via macOS VoiceProcessingIO (AEC + noise suppression built in)
2. **Silero VAD** detects speech vs silence
3. **Smart Turn v3** confirms end-of-turn on silence
4. **Moonshine** transcribes your audio to text (CPU)
5. **Gemma 4 E4B** responds using SOUL.md (+ MEMORY.md if `--memory`) as system prompt
6. **Kokoro** synthesises speech and streams sentences in parallel
7. **VPIO** delivers echo-free mic audio during TTS, enabling voice interruption

## Tech Stack

| Component | Role | Runs on |
|-----------|------|---------|
| [Moonshine](https://github.com/moonshine-ai/moonshine) | Speech-to-text | CPU |
| [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) | LLM response generation | MLX / Metal GPU |
| [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) | Text-to-speech (streaming) | CPU |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Voice activity detection | CPU |
| [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) | End-of-turn detection | CPU |
| macOS VoiceProcessingIO | Hardware AEC + noise suppression | OS kernel |
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | MLX multimodal inference | MLX / Metal GPU |

## Persona & Memory

- **SOUL.md** — Evie's personality and style. Always loaded, live-reloaded each turn. Edit it and the change takes effect on the next response.
- **MEMORY.md** — Long-term facts about you. Only active when `--memory` is passed. Evie extracts new facts after each turn and consolidates every 5 turns.

## Memory Usage

~3.5GB total. Fits comfortably on 16GB machines.

## Model Downloads

On first run, Evie downloads these models automatically:

| Model | Size | Source |
|-------|------|--------|
| Gemma 4 E4B | ~3GB | HuggingFace Hub |
| Kokoro TTS | ~300MB | GitHub releases |
| Moonshine | ~250MB | Moonshine CDN |
| Smart Turn v3 | ~60MB | HuggingFace CDN |

All models are cached locally after the first download.

## Credits

Built with open-source tools from Moonshine, Google, Kokoro, Silero, Pipecat, and the MLX community.

> Need a custom voice model or production voice agent? See [Trelis Voice AI Services](https://trelis.com/voice-ai-services/).
