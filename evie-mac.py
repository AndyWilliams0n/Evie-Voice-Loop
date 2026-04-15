#!/usr/bin/env python3
"""Voice Loop — a minimal on-device voice agent. Mac M4 / Apple Silicon.

Moonshine (CPU) transcribes speech. Gemma 4 E4B (Metal) responds.
Kokoro TTS speaks the response. macOS VoiceProcessingIO enables voice interrupt.

Usage:
    uv run evie-mac.py                        # defaults (TUI + TTS + VPIO)
    uv run evie-mac.py --no-tui               # console mode (original)
    uv run evie-mac.py --no-tts               # text out only
    uv run evie-mac.py --no-vpio              # keypress interrupt only
"""

import argparse
import os
import queue
import random
import re
import select
import sys
import tempfile
import termios
import threading
import time as _time
import tty
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np
import sounddevice as sd
sd.default.latency = 'high'
import torch

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
MAX_HISTORY = 10


def _download_file(url: str, dest: str, label: str = '', max_attempts: int = 5) -> None:
    import requests

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        headers = {'Range': f'bytes={existing}-'} if existing else {}

        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code == 416:
                    return

                r.raise_for_status()

                total = int(r.headers.get('content-length', 0)) + existing
                mode = 'ab' if existing else 'wb'
                desc = label or os.path.basename(dest)

                print(
                    f"  {'Resuming' if existing else 'Downloading'} {desc}"
                    f"{f' ({attempt}/{max_attempts})' if attempt > 1 else ''}"
                    f"{f' — {total // (1024*1024)}MB' if total else ''}...",
                    flush=True,
                )

                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            return

        except Exception as e:
            print(f"  Download error: {e}", flush=True)

            if attempt == max_attempts:
                if os.path.exists(dest):
                    os.remove(dest)

                raise RuntimeError(f"Failed to download {label or url} after {max_attempts} attempts.") from e

            print(f"  Retrying ({attempt + 1}/{max_attempts})...", flush=True)
CHIME_SR = 24000
_DIR = Path(__file__).parent

FILLERS = {
    'thinking': ["Hmm...", "That's a good question...", "Let me think..."],
    'acknowledge': ["Sure...", "Okay...", "Right...", "Well..."],
}


class VPIOMic:
    """macOS VoiceProcessingIO mic input via AVAudioEngine.
    Apple's built-in AEC + noise suppression — the OS captures the speaker
    output internally, so echo cancellation stays perfectly aligned with
    zero manual reference tracking.  Delivers 16 kHz mono float32 chunks."""

    def __init__(self, chunk_samples: int, audio_q: queue.Queue, record_buf: list | None = None):
        import AVFoundation as AVF

        self._AVF = AVF
        self._engine = AVF.AVAudioEngine.alloc().init()
        self._inp = self._engine.inputNode()
        self._chunk_samples = chunk_samples
        self._audio_q = audio_q
        self._record_buf = record_buf
        self._residual = np.empty(0, dtype=np.float32)

        ok, err = self._inp.setVoiceProcessingEnabled_error_(True, None)
        if not ok:
            raise RuntimeError(f"Failed to enable VoiceProcessingIO: {err}")

        self._native_sr = self._inp.outputFormatForBus_(0).sampleRate()

    def start(self):
        def _tap(buf, when):
            try:
                n = buf.frameLength()
                if n == 0:
                    return

                fcd = buf.floatChannelData()
                if fcd is None:
                    return

                raw = np.array(fcd[0][:n], dtype=np.float32)

                if self._native_sr != SAMPLE_RATE:
                    target_len = int(len(raw) * SAMPLE_RATE / self._native_sr)
                    raw = np.interp(
                        np.linspace(0, len(raw) - 1, target_len),
                        np.arange(len(raw)),
                        raw,
                    ).astype(np.float32)

                if self._record_buf is not None:
                    self._record_buf.append(raw.copy())

                combined = np.concatenate([self._residual, raw]) if len(self._residual) > 0 else raw
                pos = 0

                while pos + self._chunk_samples <= len(combined):
                    self._audio_q.put(combined[pos:pos + self._chunk_samples].copy())
                    pos += self._chunk_samples

                self._residual = combined[pos:].copy() if pos < len(combined) else np.empty(0, dtype=np.float32)
            except Exception:
                pass

        buf_size = int(self._native_sr * 0.032)
        self._inp.installTapOnBus_bufferSize_format_block_(0, buf_size, None, _tap)
        ok, err = self._engine.startAndReturnError_(None)
        if not ok:
            raise RuntimeError(f"AVAudioEngine failed to start: {err}")

    def stop(self):
        self._engine.stop()
        self._inp.removeTapOnBus_(0)


def load_system_prompt(include_memory: bool = False) -> str:
    names = ("SOUL.md", "MEMORY.md") if include_memory else ("SOUL.md",)
    parts = [(_DIR / n).read_text().strip() for n in names if (_DIR / n).exists()]
    return "\n\n".join(p for p in parts if p)


def _fade_tone(freq, dur, amp=0.6):
    n = int(dur * CHIME_SR)
    t = np.linspace(0, dur, n, dtype=np.float32)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    return amp * np.sin(2 * np.pi * freq * t) * env


def _silence(dur):
    return np.zeros(int(dur * CHIME_SR), dtype=np.float32)


def make_chime(duration=30.0, tick_every=1.5):
    head = np.concatenate([_fade_tone(880, 0.09), _silence(0.03), _fade_tone(1320, 0.10)])
    tick = _fade_tone(550, 0.04, amp=0.18)
    total = int(duration * CHIME_SR)
    buf = np.zeros(total, dtype=np.float32)
    buf[:len(head)] = head
    step = int(tick_every * CHIME_SR)

    for pos in range(len(head), total, step):
        end = min(pos + len(tick), total)
        buf[pos:end] = tick[:end - pos]

    return buf


def _lang_from_voice(v: str) -> str:
    prefix = v[:1] if len(v) > 1 and v[1] == '_' else ''
    return {
        'a': 'en-us', 'b': 'en-gb',
        'e': 'es', 'f': 'fr-fr', 'h': 'hi',
        'i': 'it', 'j': 'ja', 'p': 'pt-br', 'z': 'cmn',
    }.get(prefix, 'en-us')


def save_wav(audio, sr=SAMPLE_RATE):
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes())

    return path


def load_smart_turn():
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    from huggingface_hub import hf_hub_download

    print("Loading Smart Turn v3.2 model...", flush=True)

    model_path = hf_hub_download(
        repo_id="pipecat-ai/smart-turn-v3",
        filename="smart-turn-v3.2-cpu.onnx",
    )

    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise RuntimeError(f"Failed to load Smart Turn model: {e}") from e
    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def predict(audio_float32: np.ndarray) -> float:
        max_samples = 8 * SAMPLE_RATE
        audio_float32 = audio_float32[-max_samples:]
        features = extractor(
            audio_float32, sampling_rate=SAMPLE_RATE, max_length=max_samples,
            padding="max_length", return_attention_mask=False, return_tensors="np",
        )
        return float(session.run(None, {"input_features": features.input_features.astype(np.float32)})[0].flatten()[0])

    return predict


def _vad_prob(vad, chunk):
    p = vad(torch.from_numpy(chunk), SAMPLE_RATE)
    return p.item() if hasattr(p, "item") else p


# ---------------------------------------------------------------------------
# UI callback protocol — the pipeline pushes events here, agnostic of TUI vs console
# ---------------------------------------------------------------------------

class UICallback(Protocol):
    def on_loading(self, text: str) -> None: ...
    def on_state(self, state: str) -> None: ...
    def on_meter(self, rms: float, vad_prob: float) -> None: ...
    def on_heard(self, text: str) -> None: ...
    def on_response(self, text: str, first: bool) -> None: ...
    def on_system(self, text: str) -> None: ...
    def on_error(self, text: str) -> None: ...


class ConsoleUI:
    """Print-based UI — preserves original console behavior."""

    def __init__(self):
        self._speaking = False

    def on_loading(self, text: str) -> None:
        print(text, flush=True)

    def on_state(self, state: str) -> None:
        self._speaking = state == 'speaking'

    def on_meter(self, rms: float, vad_prob: float) -> None:
        pass

    def on_heard(self, text: str) -> None:
        print(f"  [{text}]")

    def on_response(self, text: str, first: bool) -> None:
        if first:
            print(f"\n> {text}", flush=True)
        else:
            print(f"> {text}", flush=True)

    def on_system(self, text: str) -> None:
        print(f"  {text}", flush=True)

    def on_error(self, text: str) -> None:
        print(f"  {text}", file=sys.stderr)


# ---------------------------------------------------------------------------
# VoicePipeline — extracted from main(), owns all audio/ML logic
# ---------------------------------------------------------------------------

class VoicePipeline:

    def __init__(self, args: argparse.Namespace, ui: UICallback):
        self.args = args
        self.ui = ui
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self.record_buf: list[np.ndarray] | None = [] if args.record else None
        self.history: list[dict] = []
        self.buf: list[np.ndarray] = []
        self.speaking = False
        self.silent_chunks = 0
        self.chime_started_at = 0.0
        self.silence_limit = max(1, int(args.silence_ms / (CHUNK_SAMPLES / SAMPLE_RATE * 1000)))
        self.console_mode = False

        self.vad = None
        self.moonshine = None
        self.model = None
        self.processor = None
        self._mlx_generate = None
        self._mlx_stream_generate = None
        self.smart_turn = None
        self.kokoro = None
        self.filler_cache: dict = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.chime_sound = make_chime() if args.chime else None
        self.vpio_mic = None
        self._sd_callback_active = False
        self._interrupt_evt = threading.Event()
        self._shutdown = False

    def load_models(self):
        self.ui.on_loading("Loading Silero VAD...")
        from silero_vad import load_silero_vad
        self.vad = load_silero_vad(onnx=True)

        self.ui.on_loading("Loading Moonshine (transcription)...")
        from moonshine_voice import Transcriber, get_model_for_language
        ms_path, ms_arch = get_model_for_language("en")
        self.moonshine = Transcriber(model_path=str(ms_path), model_arch=ms_arch)

        self.ui.on_loading(f"Loading {self.args.model} (first run downloads ~3GB)...")
        from mlx_vlm import load, generate, stream_generate
        self.model, self.processor = load(self.args.model)
        self._mlx_generate = generate
        self._mlx_stream_generate = stream_generate

        if self.args.smart_turn:
            self.smart_turn = load_smart_turn()

        if self.args.tts:
            self.ui.on_loading("Loading Kokoro TTS...")

            try:
                import espeakng_loader
                os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', espeakng_loader.get_library_path())
            except ImportError:
                import subprocess

                try:
                    prefix = subprocess.check_output(['brew', '--prefix', 'espeak-ng'], text=True).strip()
                    os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', f'{prefix}/lib/libespeak-ng.dylib')
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass

            from kokoro_onnx import Kokoro

            cache_dir = os.path.join(tempfile.gettempdir(), "kokoro_tts")
            model_file = os.path.join(cache_dir, "kokoro-v1.0.onnx")
            voices_file = os.path.join(cache_dir, "voices-v1.0.bin")

            if not os.path.exists(model_file) or not os.path.exists(voices_file):
                base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
                self.ui.on_loading("  Downloading kokoro model (~300MB)...")

                if not os.path.exists(model_file):
                    _download_file(f"{base}/kokoro-v1.0.onnx", model_file, "kokoro-v1.0.onnx")

                if not os.path.exists(voices_file):
                    _download_file(f"{base}/voices-v1.0.bin", voices_file, "voices-v1.0.bin")

            self.kokoro = Kokoro(model_file, voices_file)

        if self.args.filler and self.kokoro:
            self.ui.on_loading("  Pre-computing filler phrases...")

            for phrases in FILLERS.values():
                for text in phrases:
                    samples, sr = self.kokoro.create(
                        text, voice=self.args.voice, speed=1.0,
                        lang=_lang_from_voice(self.args.voice)
                    )
                    self.filler_cache[text] = (samples, sr)

    def setup_audio(self):
        if self.args.vpio:
            try:
                self.ui.on_loading("Loading macOS VoiceProcessingIO (AEC + noise suppression)...")
                self.vpio_mic = VPIOMic(CHUNK_SAMPLES, self.audio_q, self.record_buf)
                self.ui.on_system("[vpio ready]")
            except Exception as e:
                self.ui.on_error(f"[vpio failed: {e} — falling back to sounddevice]")

        self._sd_callback_active = self.vpio_mic is None

    def request_interrupt(self):
        self._interrupt_evt.set()

    def shutdown(self):
        self._shutdown = True

    # -- internal helpers --

    def _sd_callback(self, indata, frames, time, status):
        if not self._sd_callback_active:
            return

        if status:
            self.ui.on_error(str(status))

        chunk = indata[:, 0].copy()

        if self.record_buf is not None:
            self.record_buf.append(chunk)

        self.audio_q.put(chunk)

    def _drain_audio_q(self):
        while not self.audio_q.empty():
            self.audio_q.get_nowait()

    def _transcribe(self, audio_data):
        return " ".join(l.text for l in self.moonshine.transcribe_without_streaming(
            audio_data.tolist(), SAMPLE_RATE).lines if l.text).strip()

    def _llm_generate(self, messages, max_tokens=200, temperature=0.7, **kwargs):
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        r = self._mlx_generate(
            self.model, self.processor, prompt,
            max_tokens=max_tokens, temperature=temperature,
            repetition_penalty=1.2, verbose=False, **kwargs,
        )
        return r.text if hasattr(r, "text") else str(r)

    def _speak_tts(self, text):
        samples, sr = self.kokoro.create(
            text, voice=self.args.voice, speed=1.0,
            lang=_lang_from_voice(self.args.voice),
        )
        sd.play(samples, sr)
        sd.wait()

    def _sys_messages(self):
        sp = load_system_prompt(include_memory=self.args.memory)
        return [{"role": "system", "content": sp}] if sp else []

    def _wait_for_chime_gap(self):
        if self.chime_sound is None or self.chime_started_at == 0:
            return

        CHIME_HEAD = 0.22
        TICK_DUR = 0.04
        TICK_EVERY = 1.5
        t = _time.monotonic() - self.chime_started_at

        if t < CHIME_HEAD:
            _time.sleep(CHIME_HEAD - t)
            return

        phase = (t - CHIME_HEAD) % TICK_EVERY

        if phase < TICK_DUR:
            _time.sleep(TICK_DUR - phase + 0.005)

    def _llm_stream_generate(self, messages, max_tokens=1024, temperature=0.7, **kwargs):
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        for result in self._mlx_stream_generate(
            self.model, self.processor, prompt,
            max_tokens=max_tokens, temperature=temperature,
            repetition_penalty=1.2, **kwargs
        ):
            text = result.text if hasattr(result, 'text') else str(result)

            if text:
                yield text

    def _llm_stream_sentences(self, messages, **kwargs):
        buffer = ''

        for delta in self._llm_stream_generate(messages, **kwargs):
            buffer += delta

            while True:
                newline_pos = buffer.find('\n')

                if newline_pos >= 0:
                    sentence = buffer[:newline_pos].strip()
                    buffer = buffer[newline_pos + 1:]

                    if sentence:
                        yield sentence

                    continue

                match = re.search(r'[.!?](?:\s|$)', buffer)

                if match:
                    sentence = buffer[:match.end()].strip()
                    buffer = buffer[match.end():]

                    if sentence:
                        yield sentence

                    continue

                if len(buffer) > 300:
                    space = buffer.rfind(' ', 40, 300)

                    if space > 0:
                        sentence = buffer[:space].strip()
                        buffer = buffer[space + 1:]

                        if sentence:
                            yield sentence

                        continue

                break

        if buffer.strip():
            yield buffer.strip()

    def _pick_filler(self, transcript):
        if not transcript or not self.filler_cache:
            return None

        words = transcript.lower().strip().split()

        if len(words) <= 3:
            return None

        lower = transcript.lower().strip()

        if lower.endswith('?') or any(
            w in lower for w in ('what', 'how', 'why', 'explain', 'tell me')
        ):
            choices = FILLERS['thinking']
        else:
            choices = FILLERS['acknowledge']

        available = [f for f in choices if f in self.filler_cache]
        return random.choice(available) if available else None

    def _read_memory(self):
        mem_path = _DIR / "MEMORY.md"
        return mem_path.read_text() if mem_path.exists() else "# Memory\n"

    def _run_memory(self, prompt, max_tokens, temperature, label):
        try:
            return self._llm_generate(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
            ).strip()
        except Exception as e:
            self.ui.on_error(f"[{label} failed: {e}]")
            return None

    def _update_memory(self, heard, response):
        result = self._run_memory(
            f"Current memory:\n{self._read_memory()}\n\n"
            f"User said: {heard}\n\n"
            "Did the user state a new durable fact about themselves? "
            "If yes, output one short fact per line starting with '- '. "
            "If no, output ONLY: NONE. Do not invent facts.",
            max_tokens=60, temperature=0.2, label="memory update",
        )

        if result and "NONE" not in result.upper():
            lines = [l for l in result.splitlines() if l.strip().startswith("-")]

            if lines:
                mem_path = _DIR / "MEMORY.md"

                with open(mem_path, "a") as f:
                    f.write("\n" + "\n".join(lines) + "\n")

                self.ui.on_system(f"[memory +{len(lines)}]")

    def _consolidate_memory(self):
        mem_path = _DIR / "MEMORY.md"

        if not mem_path.exists():
            return

        result = self._run_memory(
            f"Here is a memory file about a user:\n\n{self._read_memory()}\n\n"
            "Rewrite it: merge duplicates, remove transient/session-specific "
            "items (questions asked, topics discussed, tests), keep only "
            "durable facts (identity, preferences, relationships, location, "
            "ongoing projects). Output the cleaned file, starting with '# Memory' "
            "followed by bullets starting with '- '. No explanation.",
            max_tokens=300, temperature=0.2, label="memory consolidation",
        )

        if result and result.startswith("# Memory"):
            mem_path.write_text(result + "\n")
            self.ui.on_system("[memory consolidated]")

    def _play_tts_streamed(self, sentence_q, interrupted_evt, filler_text=None):
        self._drain_audio_q()
        out_stream = None
        interrupted = False
        consec_speech = 0
        play_start = 0.0

        def check_barge_in():
            nonlocal consec_speech

            if not self.vpio_mic or play_start == 0.0:
                return False

            if _time.monotonic() - play_start < 0.5:
                return False

            while not self.audio_q.empty():
                mic_chunk = self.audio_q.get_nowait()

                if len(mic_chunk) < CHUNK_SAMPLES:
                    continue

                if _vad_prob(self.vad, mic_chunk.astype(np.float32)) > 0.7:
                    consec_speech += 1

                    if consec_speech >= 6:
                        return True
                else:
                    consec_speech = 0

            return False

        def _start_stream(sr):
            nonlocal out_stream, play_start

            if out_stream is not None:
                return

            if self.chime_sound is not None:
                self._wait_for_chime_gap()
                sd.stop()

            out_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            out_stream.start()
            self._drain_audio_q()
            self.vad.reset_states()
            play_start = _time.monotonic()

        def _write_audio(data, sr):
            nonlocal interrupted
            _start_stream(sr)
            shaped = data.reshape(-1, 1) if data.ndim == 1 else data

            for i in range(0, len(shaped), 4096):
                chunk = shaped[i:i + 4096]

                if interrupted_evt.is_set():
                    return True

                if self.console_mode:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.read(1)
                        interrupted = True
                        interrupted_evt.set()
                        return True

                if check_barge_in():
                    interrupted = True
                    interrupted_evt.set()
                    self.ui.on_system("[voice interrupt]")
                    return True

                out_stream.write(chunk)

            return False

        self.ui.on_state('speaking')

        try:
            if filler_text and filler_text in self.filler_cache:
                filler_samples, filler_sr = self.filler_cache[filler_text]

                if _write_audio(filler_samples, filler_sr):
                    return

                gap = np.zeros(int(0.15 * filler_sr), dtype=np.float32)

                if _write_audio(gap, filler_sr):
                    return

            while not interrupted_evt.is_set():
                try:
                    sentence = sentence_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                if sentence is None:
                    break

                consec_speech = 0

                samples, sr = self.kokoro.create(
                    sentence, voice=self.args.voice, speed=1.0,
                    lang=_lang_from_voice(self.args.voice)
                )

                if _write_audio(samples, sr):
                    break

                if not interrupted_evt.is_set() and out_stream and self.args.sentence_gap_ms > 0:
                    gap_samples = int(self.args.sentence_gap_ms / 1000 * out_stream.samplerate)

                    if gap_samples > 0:
                        gap = np.zeros(gap_samples, dtype=np.float32)

                        if _write_audio(gap, int(out_stream.samplerate)):
                            break

        except Exception as e:
            self.ui.on_error(f"[TTS error: {e}]")
            interrupted_evt.set()

        finally:
            if out_stream:
                out_stream.stop()
                out_stream.close()

            if interrupted:
                self.ui.on_system("[interrupted]")

            self._drain_audio_q()
            self.vad.reset_states()
            self.ui.on_state('listening')

    def _process_utterance(self, audio):
        self.ui.on_system(f"({len(audio) / SAMPLE_RATE:.1f}s)")

        if self.chime_sound is not None:
            sd.play(self.chime_sound, CHIME_SR)
            self.chime_started_at = _time.monotonic()

        wav_path = save_wav(audio) if self.args.audio_mode else None

        try:
            messages = self._sys_messages()

            for h in self.history[-MAX_HISTORY:]:
                messages += [
                    {"role": "user", "content": h["user"]},
                    {"role": "assistant", "content": h["assistant"]},
                ]

            if self.args.audio_mode:
                transcribe_future = self.executor.submit(self._transcribe, audio)
                messages.append({"role": "user", "content": [{"type": "audio"}]})
            else:
                self.ui.on_state('thinking')
                heard = self._transcribe(audio)
                self.ui.on_heard(heard)

                if not heard:
                    self.ui.on_state('listening')
                    return

                messages.append({"role": "user", "content": heard})

            gen_kwargs = {"audio": [wav_path]} if self.args.audio_mode else {}

            if self.kokoro:
                sentence_q = queue.Queue()
                interrupted_evt = threading.Event()
                self._interrupt_evt = interrupted_evt
                filler_text = None

                if self.args.filler and not self.args.audio_mode:
                    filler_text = self._pick_filler(heard)

                tts_thread = threading.Thread(
                    target=self._play_tts_streamed,
                    args=(sentence_q, interrupted_evt, filler_text),
                    daemon=True,
                )
                tts_thread.start()

                if filler_text:
                    self.ui.on_response(filler_text, first=True)

                response_parts = []
                first_output = filler_text is not None

                try:
                    for sentence in self._llm_stream_sentences(messages, **gen_kwargs):
                        if interrupted_evt.is_set():
                            break

                        response_parts.append(sentence)
                        sentence_q.put(sentence)
                        self.ui.on_response(sentence, first=not first_output)
                        first_output = True

                except Exception as e:
                    self.ui.on_error(f"[stream error: {e}]")

                    if not response_parts:
                        response = self._llm_generate(messages, **gen_kwargs)
                        response_parts = [response]
                        sentence_q.put(response)
                        self.ui.on_response(response, first=True)

                finally:
                    sentence_q.put(None)

                tts_thread.join(timeout=30)
                response = ' '.join(response_parts)

                if self.args.audio_mode:
                    heard = transcribe_future.result(timeout=10)
                    self.ui.on_heard(heard)

            else:
                self.ui.on_state('thinking')
                response = self._llm_generate(messages, **gen_kwargs)

                if self.args.audio_mode:
                    heard = transcribe_future.result(timeout=10)
                    self.ui.on_heard(heard)

                self.ui.on_response(response, first=True)

                if self.chime_sound is not None:
                    self._wait_for_chime_gap()
                    sd.stop()

                self.ui.on_state('listening')

            self.history.append({"user": heard, "assistant": response})

            if len(self.history) > MAX_HISTORY:
                self.history.pop(0)

            if self.args.memory:
                self._update_memory(heard, response)

                if len(self.history) % 5 == 0:
                    self._consolidate_memory()

        except Exception as e:
            self.ui.on_error(f"Error: {e}")
            self.ui.on_state('listening')

        finally:
            if wav_path:
                os.unlink(wav_path)

    def run_loop(self):
        """Main voice loop. Blocks until shutdown or KeyboardInterrupt.
        Call load_models() and setup_audio() before this."""
        greeting = self._llm_generate(self._sys_messages() + [
            {"role": "user", "content": (
                "Greet the user as Eevie, thats your name, in one short sentence. "
                "If my name is in memory, use it and ask how you can help. "
                "Otherwise, ask for my name."
            )},
        ], max_tokens=60)
        self.ui.on_response(greeting, first=True)

        if self.kokoro:
            self._speak_tts(greeting)

        if self.vpio_mic:
            self.vpio_mic.start()

        self.ui.on_state('listening')

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=CHUNK_SAMPLES, callback=self._sd_callback,
        ):
            try:
                while not self._shutdown:
                    try:
                        chunk = self.audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if len(chunk) < CHUNK_SAMPLES:
                        continue

                    speech_prob = _vad_prob(self.vad, chunk)
                    mic_rms = np.sqrt(np.mean(chunk ** 2))
                    self.ui.on_meter(mic_rms, speech_prob)

                    if speech_prob > 0.5:
                        if not self.speaking:
                            self.speaking = True

                        self.silent_chunks = 0
                        self.buf.append(chunk)

                    elif self.speaking:
                        self.silent_chunks += 1
                        self.buf.append(chunk)

                        if self.silent_chunks < self.silence_limit:
                            continue

                        if self.smart_turn and self.buf:
                            prob = self.smart_turn(np.concatenate(self.buf))

                            if prob < 0.5:
                                self.silent_chunks = 0
                                continue

                        self._process_utterance(np.concatenate(self.buf))
                        self.buf.clear()
                        self.speaking = False
                        self.silent_chunks = 0
                        self.vad.reset_states()

            except KeyboardInterrupt:
                pass

            finally:
                if self.vpio_mic:
                    self.vpio_mic.stop()

                self.executor.shutdown(wait=False)

                if self.args.record and self.record_buf:
                    full = np.concatenate(self.record_buf)

                    with wave.open(self.args.record, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes((full * 32767).clip(-32768, 32767).astype(np.int16).tobytes())

                    self.ui.on_system(f"Recorded {len(full) / SAMPLE_RATE:.1f}s to {self.args.record}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evie — your personal voice assistant (Mac)")
    B = argparse.BooleanOptionalAction
    ap.add_argument("--tts", action=B, default=True, help="Kokoro TTS output")
    ap.add_argument("--smart-turn", action=B, default=True, help="Smart Turn v3 endpoint detection")
    ap.add_argument("--vpio", action=B, default=True,
                    help="macOS VoiceProcessingIO echo cancellation (voice interrupt)")
    ap.add_argument("--chime", action=B, default=False,
                    help="Chime on utterance + soft ticks while generating")
    ap.add_argument("--memory", action="store_true",
                    help="Read/write MEMORY.md (auto-update durable facts, consolidate every 5 turns)")
    ap.add_argument("--filler", action=B, default=True,
                    help="Filler phrases during LLM prefill")
    ap.add_argument("--sentence-gap-ms", type=int, default=80,
                    help="Natural gap between streamed sentences (ms)")
    ap.add_argument("--audio-mode", action="store_true", help="Send audio directly to Gemma (experimental)")
    ap.add_argument("--model", default="mlx-community/gemma-4-E4B-it-4bit")
    ap.add_argument("--silence-ms", type=int, default=700)
    ap.add_argument("--record", nargs="?", const="", metavar="FILE",
                    help="Record mic to WAV for debugging (default: tmp/recording-TIMESTAMP.wav)")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice")
    ap.add_argument("--tui", action=B, default=True, help="Textual TUI (default on)")
    args = ap.parse_args()

    if args.record == "":
        tmp_dir = _DIR / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        args.record = str(tmp_dir / f"recording-{_time.strftime('%Y%m%d-%H%M%S')}.wav")

    # Load models before entering TUI (subprocesses need real stdout/stderr)
    console_ui = ConsoleUI()
    pipeline = VoicePipeline(args, console_ui)
    pipeline.load_models()
    pipeline.setup_audio()

    if args.tui:
        try:
            from evie_tui import EvieTUI
            app = EvieTUI(args, pipeline)
            app.run()
            return
        except ImportError:
            print("Textual not installed — falling back to console mode", file=sys.stderr)

    # Console mode
    pipeline.console_mode = True

    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        pipeline.run_loop()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)


if __name__ == "__main__":
    main()
