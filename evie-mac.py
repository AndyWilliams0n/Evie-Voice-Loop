#!/usr/bin/env python3
"""Voice Loop — a minimal on-device voice agent. Mac M4 / Apple Silicon.

Moonshine (CPU) transcribes speech. Gemma 4 E4B (Metal) responds.
Kokoro TTS speaks the response. macOS VoiceProcessingIO enables voice interrupt.

Usage:
    uv run voice_loop_mac.py                        # defaults (TTS + smart turn + VPIO)
    uv run voice_loop_mac.py --no-tts               # text out only
    uv run voice_loop_mac.py --no-vpio              # keypress interrupt only
    uv run voice_loop_mac.py --chime-loop           # chime + ticks while generating
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

import numpy as np
import sounddevice as sd
sd.default.latency = 'high'
import torch

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz (required by Silero VAD)
MAX_HISTORY = 10
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
    """Tone with raised-cosine (Hann) envelope — smooth fade in/out, no clicks."""
    n = int(dur * CHIME_SR)
    t = np.linspace(0, dur, n, dtype=np.float32)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    return amp * np.sin(2 * np.pi * freq * t) * env

def _silence(dur):
    return np.zeros(int(dur * CHIME_SR), dtype=np.float32)

def make_chime(duration=30.0, tick_every=1.5):
    """Two-tone chime + periodic short ticks. Single buffer → one sd.play()."""
    head = np.concatenate([_fade_tone(880, 0.09), _silence(0.03), _fade_tone(1320, 0.10)])
    # Short soft click-style tick (shorter and quieter than a beep)
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
    """Infer Kokoro lang code from voice prefix.
    a* = US English, b* = UK English, e* = Spanish, f* = French,
    h* = Hindi, i* = Italian, j* = Japanese, p* = Portuguese, z* = Chinese."""
    prefix = v[:1] if len(v) > 1 and v[1] == '_' else ''
    return {
        'a': 'en-us', 'b': 'en-gb',
        'e': 'es', 'f': 'fr-fr', 'h': 'hi',
        'i': 'it', 'j': 'ja', 'p': 'pt-br', 'z': 'cmn',
    }.get(prefix, 'en-us')


def save_wav(audio, sr=SAMPLE_RATE):
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    return path


def load_smart_turn():
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    model_path = os.path.join(tempfile.gettempdir(), "smart_turn_v3", "smart_turn_v3.2_cpu.onnx")
    if not os.path.exists(model_path):
        print("Downloading Smart Turn v3.2 model...", flush=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx", model_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
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
    args = ap.parse_args()
    if args.record == "":
        tmp_dir = _DIR / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        args.record = str(tmp_dir / f"recording-{_time.strftime('%Y%m%d-%H%M%S')}.wav")
    silence_limit = max(1, int(args.silence_ms / (CHUNK_SAMPLES / SAMPLE_RATE * 1000)))

    print("Loading Silero VAD...", flush=True)
    from silero_vad import load_silero_vad
    vad = load_silero_vad(onnx=True)
    print("Loading Moonshine (transcription)...", flush=True)
    from moonshine_voice import Transcriber, get_model_for_language
    ms_path, ms_arch = get_model_for_language("en")
    moonshine = Transcriber(model_path=str(ms_path), model_arch=ms_arch)
    print(f"Loading {args.model} (first run downloads ~3GB)...", flush=True)
    from mlx_vlm import load, generate, stream_generate
    model, processor = load(args.model)
    smart_turn = load_smart_turn() if args.smart_turn else None
    kokoro = None
    if args.tts:
        print("Loading Kokoro TTS...", flush=True)
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
        if not os.path.exists(model_file):
            os.makedirs(cache_dir, exist_ok=True)
            import urllib.request
            base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
            print("  Downloading kokoro model (~300MB)...", flush=True)
            urllib.request.urlretrieve(f"{base}/kokoro-v1.0.onnx", model_file)
            urllib.request.urlretrieve(f"{base}/voices-v1.0.bin", voices_file)
        kokoro = Kokoro(model_file, voices_file)

    filler_cache = {}
    if args.filler and kokoro:
        print("  Pre-computing filler phrases...", flush=True)
        for phrases in FILLERS.values():
            for text in phrases:
                samples, sr = kokoro.create(
                    text, voice=args.voice, speed=1.0,
                    lang=_lang_from_voice(args.voice)
                )
                filler_cache[text] = (samples, sr)

    executor = ThreadPoolExecutor(max_workers=1)
    chime_sound = make_chime() if args.chime else None
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    record_buf: list[np.ndarray] | None = [] if args.record else None

    vpio_mic = None
    if args.vpio:
        try:
            print("Loading macOS VoiceProcessingIO (AEC + noise suppression)...", flush=True)
            vpio_mic = VPIOMic(CHUNK_SAMPLES, audio_q, record_buf)
            print("  VPIO ready", flush=True)
        except Exception as e:
            print(f"  VPIO failed: {e} — falling back to sounddevice", file=sys.stderr)

    sd_callback_active = vpio_mic is None

    def callback(indata, frames, time, status):
        if not sd_callback_active:
            return

        if status:
            print(status, file=sys.stderr)

        chunk = indata[:, 0].copy()

        if record_buf is not None:
            record_buf.append(chunk)

        audio_q.put(chunk)

    def drain_audio_q():
        while not audio_q.empty():
            audio_q.get_nowait()

    def transcribe(audio_data):
        return " ".join(l.text for l in moonshine.transcribe_without_streaming(
            audio_data.tolist(), SAMPLE_RATE).lines if l.text).strip()

    def llm_generate(messages, max_tokens=200, temperature=0.7, **kwargs):
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        r = generate(model, processor, prompt, max_tokens=max_tokens,
                     temperature=temperature, repetition_penalty=1.2, verbose=False, **kwargs)
        return r.text if hasattr(r, "text") else str(r)

    def speak_tts(text):
        samples, sr = kokoro.create(text, voice=args.voice, speed=1.0, lang=_lang_from_voice(args.voice))
        sd.play(samples, sr); sd.wait()

    _mem_path = _DIR / "MEMORY.md"

    def _read_memory():
        return _mem_path.read_text() if _mem_path.exists() else "# Memory\n"

    def _run_memory(prompt, max_tokens, temperature, label):
        try:
            return llm_generate(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
            ).strip()
        except Exception as e:
            print(f"  [{label} failed: {e}]", file=sys.stderr)
            return None

    def update_memory(heard, response):
        result = _run_memory(
            f"Current memory:\n{_read_memory()}\n\n"
            f"User said: {heard}\n\n"
            "Did the user state a new durable fact about themselves? "
            "If yes, output one short fact per line starting with '- '. "
            "If no, output ONLY: NONE. Do not invent facts.",
            max_tokens=60, temperature=0.2, label="memory update",
        )
        if result and "NONE" not in result.upper():
            lines = [l for l in result.splitlines() if l.strip().startswith("-")]
            if lines:
                with open(_mem_path, "a") as f:
                    f.write("\n" + "\n".join(lines) + "\n")
                print(f"  [memory +{len(lines)}]", flush=True)

    def consolidate_memory():
        if not _mem_path.exists():
            return
        result = _run_memory(
            f"Here is a memory file about a user:\n\n{_read_memory()}\n\n"
            "Rewrite it: merge duplicates, remove transient/session-specific "
            "items (questions asked, topics discussed, tests), keep only "
            "durable facts (identity, preferences, relationships, location, "
            "ongoing projects). Output the cleaned file, starting with '# Memory' "
            "followed by bullets starting with '- '. No explanation.",
            max_tokens=300, temperature=0.2, label="memory consolidation",
        )
        if result and result.startswith("# Memory"):
            _mem_path.write_text(result + "\n")
            print("  [memory consolidated]", flush=True)

    def _sys_messages():
        sp = load_system_prompt(include_memory=args.memory)
        return [{"role": "system", "content": sp}] if sp else []

    def _wait_for_chime_gap():
        """Wait until we're in a silent gap between ticks, so sd.stop() doesn't
        clip a tick mid-cycle (which clicks). Max wait ~40ms."""
        if chime_sound is None or chime_started_at[0] == 0:
            return
        CHIME_HEAD = 0.22  # end of chime tones in buffer
        TICK_DUR = 0.04    # tick length
        TICK_EVERY = 1.5
        t = _time.monotonic() - chime_started_at[0]
        if t < CHIME_HEAD:
            # Still in chime head; wait for end of chime then it's safe
            _time.sleep(CHIME_HEAD - t)
            return
        phase = (t - CHIME_HEAD) % TICK_EVERY
        if phase < TICK_DUR:
            # In a tick — wait until it ends
            _time.sleep(TICK_DUR - phase + 0.005)

    def llm_stream_generate(messages, max_tokens=200, temperature=0.7, **kwargs):
        """Yield text deltas from the LLM, token by token.
        Note: stream_generate's result.text is already a delta
        (the detokenizer tracks offset internally via last_segment)."""
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for result in stream_generate(
            model, processor, prompt,
            max_tokens=max_tokens, temperature=temperature,
            repetition_penalty=1.2, **kwargs
        ):
            text = result.text if hasattr(result, 'text') else str(result)
            if text:
                yield text

    def llm_stream_sentences(messages, **kwargs):
        """Buffer LLM token deltas into complete sentences."""
        buffer = ''
        for delta in llm_stream_generate(messages, **kwargs):
            buffer += delta
            while True:
                match = re.search(r'[.!?](?:\s|$)', buffer)
                if match:
                    sentence = buffer[:match.end()].strip()
                    buffer = buffer[match.end():]
                    if sentence:
                        yield sentence
                    continue

                if len(buffer) > 120:
                    comma = buffer.find(', ', 40)
                    if comma > 0:
                        sentence = buffer[:comma + 1].strip()
                        buffer = buffer[comma + 2:]
                        if sentence:
                            yield sentence
                        continue

                if len(buffer) > 200:
                    space = buffer.rfind(' ', 40, 200)
                    if space > 0:
                        sentence = buffer[:space].strip()
                        buffer = buffer[space + 1:]
                        if sentence:
                            yield sentence
                        continue

                break

        if buffer.strip():
            yield buffer.strip()

    def pick_filler(transcript):
        """Choose an appropriate filler based on the user's utterance."""
        if not transcript or not filler_cache:
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

        available = [f for f in choices if f in filler_cache]
        return random.choice(available) if available else None

    def play_tts_streamed(sentence_q, interrupted_evt, filler_text=None):
        """TTS worker: play filler, then stream sentences one at a time.
        Barge-in uses VPIO-cleaned mic audio (echo already removed by OS)
        so detection is just VAD — no manual reference tracking needed."""
        drain_audio_q()
        out_stream, interrupted = None, False
        consec_speech = 0
        play_start = 0.0

        def check_barge_in():
            nonlocal consec_speech
            if not vpio_mic or play_start == 0.0:
                return False

            if _time.monotonic() - play_start < 0.5:
                return False

            while not audio_q.empty():
                mic_chunk = audio_q.get_nowait()

                if len(mic_chunk) < CHUNK_SAMPLES:
                    continue

                if _vad_prob(vad, mic_chunk.astype(np.float32)) > 0.7:
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

            if chime_sound is not None:
                _wait_for_chime_gap()
                sd.stop()

            out_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            out_stream.start()
            drain_audio_q()
            vad.reset_states()
            play_start = _time.monotonic()

        def _write_audio(data, sr):
            nonlocal interrupted
            _start_stream(sr)
            shaped = data.reshape(-1, 1) if data.ndim == 1 else data

            for i in range(0, len(shaped), 4096):
                chunk = shaped[i:i + 4096]

                if interrupted_evt.is_set():
                    return True

                if select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.read(1)
                    interrupted = True
                    interrupted_evt.set()
                    return True

                if check_barge_in():
                    interrupted = True
                    interrupted_evt.set()
                    print("  [voice interrupt]", flush=True)
                    return True

                out_stream.write(chunk)

            return False

        try:
            if filler_text and filler_text in filler_cache:
                filler_samples, filler_sr = filler_cache[filler_text]

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

                samples, sr = kokoro.create(
                    sentence, voice=args.voice, speed=1.0,
                    lang=_lang_from_voice(args.voice)
                )

                if _write_audio(samples, sr):
                    break

                if not interrupted_evt.is_set() and out_stream and args.sentence_gap_ms > 0:
                    gap_samples = int(args.sentence_gap_ms / 1000 * out_stream.samplerate)

                    if gap_samples > 0:
                        gap = np.zeros(gap_samples, dtype=np.float32)

                        if _write_audio(gap, int(out_stream.samplerate)):
                            break

        except Exception as e:
            print(f"  [TTS error: {e}]", file=sys.stderr)
            interrupted_evt.set()

        finally:
            if out_stream:
                out_stream.stop()
                out_stream.close()

            if interrupted:
                print("  [interrupted]")

            drain_audio_q()
            vad.reset_states()

    def process_utterance(audio, history):
        print(f" ({len(audio) / SAMPLE_RATE:.1f}s)")
        if chime_sound is not None:
            print("  *chime*", flush=True)
            sd.play(chime_sound, CHIME_SR)
            chime_started_at[0] = _time.monotonic()

        wav_path = save_wav(audio) if args.audio_mode else None

        try:
            messages = _sys_messages()
            for h in history[-MAX_HISTORY:]:
                messages += [
                    {"role": "user", "content": h["user"]},
                    {"role": "assistant", "content": h["assistant"]},
                ]

            if args.audio_mode:
                transcribe_future = executor.submit(transcribe, audio)
                messages.append({"role": "user", "content": [{"type": "audio"}]})
            else:
                heard = transcribe(audio)
                print(f"  [{heard}]")
                if not heard:
                    return
                messages.append({"role": "user", "content": heard})

            gen_kwargs = {"audio": [wav_path]} if args.audio_mode else {}

            if kokoro:
                # Pipelined: LLM streams sentences while TTS plays them
                sentence_q = queue.Queue()
                interrupted_evt = threading.Event()
                filler_text = None

                if args.filler and not args.audio_mode:
                    filler_text = pick_filler(heard)

                tts_thread = threading.Thread(
                    target=play_tts_streamed,
                    args=(sentence_q, interrupted_evt, filler_text),
                    daemon=True,
                )
                tts_thread.start()

                if filler_text:
                    print(f"\n> {filler_text}", flush=True)

                response_parts = []
                first_output = filler_text is not None

                try:
                    for sentence in llm_stream_sentences(messages, **gen_kwargs):
                        if interrupted_evt.is_set():
                            break

                        response_parts.append(sentence)
                        sentence_q.put(sentence)

                        if first_output:
                            print(f"> {sentence}", flush=True)
                        else:
                            print(f"\n> {sentence}", flush=True)
                            first_output = True

                except Exception as e:
                    print(f"  [stream error: {e}]", file=sys.stderr)

                    if not response_parts:
                        # Fallback to full generation
                        response = llm_generate(messages, **gen_kwargs)
                        response_parts = [response]
                        sentence_q.put(response)
                        print(f"\n> {response}", flush=True)

                finally:
                    sentence_q.put(None)

                tts_thread.join(timeout=30)
                response = ' '.join(response_parts)
                print(flush=True)

                if args.audio_mode:
                    heard = transcribe_future.result(timeout=10)
                    print(f"  [{heard}]")

            else:
                # No TTS: sequential generation
                response = llm_generate(messages, **gen_kwargs)

                if args.audio_mode:
                    heard = transcribe_future.result(timeout=10)
                    print(f"  [{heard}]")

                print(f"\n> {response}\n", flush=True)

                if chime_sound is not None:
                    _wait_for_chime_gap()
                    sd.stop()

            history.append({"user": heard, "assistant": response})

            if len(history) > MAX_HISTORY:
                history.pop(0)

            if args.memory:
                update_memory(heard, response)

                if len(history) % 5 == 0:
                    consolidate_memory()

        except Exception as e:
            print(f"\nError: {e}\n", file=sys.stderr)

        finally:
            if wav_path:
                os.unlink(wav_path)

    history, buf = [], []
    chime_started_at = [0.0]  # monotonic time when last chime started (for tick-boundary TTS start)
    speaking, silent_chunks = False, 0

    # Set terminal to raw mode so keypress interrupts work without Enter
    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    mode = "audio" if args.audio_mode else "text"
    aec_label = "vpio" if vpio_mic else "off"
    print(f"\nListening (mode: {mode}, tts: {args.tts}, silence: {args.silence_ms}ms, smart-turn: {args.smart_turn}, aec: {aec_label})")
    tts_hint = (" Speak or press any key to interrupt TTS." if vpio_mic else " Press any key to interrupt TTS.") if args.tts else ""
    print(f"Speak into your microphone. Ctrl+C to quit.{tts_hint}\n", flush=True)

    greeting = llm_generate(_sys_messages() + [
        {"role": "user", "content": (
            "Greet the user as Eevie, thats your name, in one short sentence. "
            "If my name is in memory, use it and ask how you can help. "
            "Otherwise, ask for my name."
        )},
    ], max_tokens=60)
    print(f"> {greeting}\n", flush=True)
    if kokoro:
        speak_tts(greeting)

    if vpio_mic:
        vpio_mic.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=CHUNK_SAMPLES, callback=callback,
    ):
        try:
            while True:
                chunk = audio_q.get()

                if len(chunk) < CHUNK_SAMPLES:
                    continue

                speech_prob = _vad_prob(vad, chunk)
                mic_rms = np.sqrt(np.mean(chunk ** 2))

                if speech_prob > 0.5:
                    if not speaking:
                        speaking = True
                        print(f"[listening rms={mic_rms:.3f}...]", end="", flush=True)

                    silent_chunks = 0
                    buf.append(chunk)

                elif speaking:
                    silent_chunks += 1
                    buf.append(chunk)

                    if silent_chunks < silence_limit:
                        continue

                    if smart_turn and buf:
                        prob = smart_turn(np.concatenate(buf))
                        print(f" [turn prob: {prob:.2f}]", end="", flush=True)

                        if prob < 0.5:
                            silent_chunks = 0
                            continue

                    process_utterance(np.concatenate(buf), history)
                    buf.clear()
                    speaking, silent_chunks = False, 0
                    vad.reset_states()

        except KeyboardInterrupt:
            print("\nBye!")
            executor.shutdown(wait=False)

        finally:
            if vpio_mic:
                vpio_mic.stop()

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

            if args.record and record_buf:
                full = np.concatenate(record_buf)

                with wave.open(args.record, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes((full * 32767).clip(-32768, 32767).astype(np.int16).tobytes())

                print(f"Recorded {len(full) / SAMPLE_RATE:.1f}s to {args.record}", flush=True)


if __name__ == "__main__":
    main()
