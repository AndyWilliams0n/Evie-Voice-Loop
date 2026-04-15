# Meet Evie

Evie is your personal voice assistant that runs entirely on your Mac. No cloud, no API keys, no subscriptions. Just you and Evie, having a conversation.

She listens, thinks, and speaks back — all powered by your Apple Silicon chip.

## Get Started

```bash
uv sync
uv run evie-mac.py
```

That's it. The first time you run Evie, she'll download a few models (~3.5GB total). After that, everything runs locally and offline.

## Talking to Evie

Just speak naturally. Evie will wait for you to finish your sentence before responding. If you want to interrupt her, just start talking — she'll stop and listen.

You can also press any key to interrupt her mid-sentence.

## Make Evie Your Own

Evie's personality lives in a simple text file called `SOUL.md`. Open it, change it, and Evie picks up the changes on her very next response. No restart needed.

Want Evie to remember things about you between conversations? Start her with memory enabled:

```bash
uv run evie-mac.py --memory
```

She'll learn facts about you over time and store them in `MEMORY.md`.

## Pick a Voice

Evie defaults to a US female voice, but you can change it:

```bash
uv run evie-mac.py --voice bf_emma
```

| Voice | Accent | Gender |
|-------|--------|--------|
| `af_heart` | US | Female (default) |
| `af_bella` | US | Female |
| `bf_emma` | UK | Female |
| `am_fenrir` | US | Male |
| `am_puck` | US | Male |
| `bm_george` | UK | Male |

## Common Options

| Flag | What it does |
|------|-------------|
| `--memory` | Evie remembers things about you between sessions |
| `--chime` | Play a chime when Evie hears you, with soft ticks while she thinks |
| `--no-tts` | Text-only mode (no voice output) |
| `--voice NAME` | Change Evie's voice |

For the full list of options and technical details, see [README.advanced.md](README.advanced.md).

## Requirements

- Mac with Apple Silicon (M1/M2/M3/M4)
- ~3.5GB disk space for models
- ~3.5GB RAM while running
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)

## License

Apache 2.0.
