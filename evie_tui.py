"""Textual TUI for Evie voice assistant.

Provides: status bar, live audio meter with peak hold,
scrolling conversation log, and keypress interrupt handling.
"""

import argparse
import threading
import time as _time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, RichLog, Static
from textual.widget import Widget


# ---------------------------------------------------------------------------
# AudioMeter — live RMS bar with peak hold
# ---------------------------------------------------------------------------

class AudioMeter(Widget):
    """Single-line audio level meter with peak hold and VAD colouring."""

    DEFAULT_CSS = """
    AudioMeter {
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._rms = 0.0
        self._peak = 0.0
        self._peak_time = 0.0
        self._vad_active = False
        self._max_rms = 0.4

    def update_level(self, rms: float, vad_prob: float) -> None:
        now = _time.monotonic()
        self._rms = rms
        self._vad_active = vad_prob > 0.5

        if rms > self._peak:
            self._peak = rms
            self._peak_time = now

        held = now - self._peak_time
        if held > 1.0:
            decay = (held - 1.0) * 0.8
            self._peak = max(rms, self._peak - decay)

        self.refresh()

    def render(self) -> Text:
        w = self.size.width - 2
        label = f" {self._rms:.3f}  peak {self._peak:.3f}"
        bar_w = max(1, w - len(label))

        norm_rms = min(self._rms / self._max_rms, 1.0)
        norm_peak = min(self._peak / self._max_rms, 1.0)
        filled = int(norm_rms * bar_w)
        peak_pos = int(norm_peak * bar_w)
        peak_pos = min(peak_pos, bar_w - 1)

        bar_color = "green" if self._vad_active else "grey62"
        result = Text()

        for i in range(bar_w):
            if i == peak_pos and self._peak > 0.001:
                result.append("│", style="bold yellow")
            elif i < filled:
                result.append("█", style=bar_color)
            else:
                result.append("░", style="grey30")

        result.append(label, style="grey70")
        return result


# ---------------------------------------------------------------------------
# StatusBar — current state + config summary
# ---------------------------------------------------------------------------

class StatusBar(Static):
    """Top-line status: state indicator + config flags."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 1;
        background: $surface;
    }
    """

    STATE_STYLES = {
        'listening': ('● LISTENING', 'bold green'),
        'thinking': ('● THINKING', 'bold yellow'),
        'speaking': ('● SPEAKING', 'bold dodger_blue1'),
        'loading': ('● LOADING', 'bold grey70'),
    }

    def __init__(self, config_text: str = '') -> None:
        super().__init__()
        self._state = 'loading'
        self._config_text = config_text

    def set_state(self, state: str) -> None:
        self._state = state
        self._render_bar()

    def _render_bar(self) -> None:
        label, style = self.STATE_STYLES.get(self._state, ('● ...', 'grey70'))
        text = Text()
        text.append(label, style=style)

        if self._config_text:
            padding = '  '
            text.append(padding)
            text.append(self._config_text, style="grey50")

        self.update(text)


# ---------------------------------------------------------------------------
# TUICallback — bridges VoicePipeline events to Textual widgets
# ---------------------------------------------------------------------------

class TUICallback:
    """UICallback implementation that pushes updates to the TUI via call_from_thread."""

    def __init__(self, app: 'EvieTUI') -> None:
        self._app = app
        self._last_meter = 0.0

    def on_loading(self, text: str) -> None:
        self._app.call_from_thread(self._app.log_system, text)

    def on_state(self, state: str) -> None:
        self._app.call_from_thread(self._app.set_state, state)

    def on_meter(self, rms: float, vad_prob: float) -> None:
        now = _time.monotonic()

        if now - self._last_meter < 0.033:
            return

        self._last_meter = now
        self._app.call_from_thread(self._app.update_meter, rms, vad_prob)

    def on_heard(self, text: str) -> None:
        self._app.call_from_thread(self._app.log_heard, text)

    def on_response(self, text: str, first: bool) -> None:
        self._app.call_from_thread(self._app.log_response, text)

    def on_system(self, text: str) -> None:
        self._app.call_from_thread(self._app.log_system, text)

    def on_error(self, text: str) -> None:
        self._app.call_from_thread(self._app.log_error, text)


# ---------------------------------------------------------------------------
# EvieTUI — the main Textual application
# ---------------------------------------------------------------------------

class EvieTUI(App):
    """Evie voice assistant terminal UI."""

    TITLE = 'Evie'
    SUB_TITLE = 'Voice Assistant'

    CSS = """
    Screen {
        layout: vertical;
    }

    Header {
        dock: top;
        height: 1;
    }

    StatusBar {
        dock: top;
    }

    AudioMeter {
        dock: top;
    }

    #conversation {
        height: 1fr;
        scrollbar-size: 1 1;
        border-top: solid $surface-lighten-1;
        border-bottom: solid $surface-lighten-1;
    }

    Footer {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "interrupt", "Interrupt"),
    ]

    def __init__(self, args: argparse.Namespace, pipeline) -> None:
        super().__init__()
        self._args = args
        self._pipeline = pipeline
        self._pipeline_thread = None
        self._state = 'loading'

    def compose(self) -> ComposeResult:
        config_parts = []

        if self._args.vpio:
            config_parts.append('vpio:on')

        config_parts.append(f'tts:{"on" if self._args.tts else "off"}')
        model_short = self._args.model.split('/')[-1].split('-')[1] if '/' in self._args.model else self._args.model
        config_parts.append(f'model:{model_short}')

        yield Header()
        yield StatusBar(config_text='  '.join(config_parts))
        yield AudioMeter()
        yield RichLog(markup=True, highlight=False, max_lines=500, id="conversation")
        yield Footer()

    def on_mount(self) -> None:
        self._pipeline.ui = TUICallback(self)

        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline,
            daemon=True,
        )
        self._pipeline_thread.start()

    def _run_pipeline(self) -> None:
        try:
            self._pipeline.run_loop()
        except Exception as e:
            self.call_from_thread(self.log_error, f"Pipeline crashed: {e}")

    # -- actions --

    def action_interrupt(self) -> None:
        if self._state == 'speaking' and self._pipeline:
            self._pipeline.request_interrupt()

    def action_quit(self) -> None:
        if self._pipeline:
            self._pipeline.shutdown()

        self.exit()

    def on_key(self, event) -> None:
        if self._state == 'speaking' and event.key not in ('q', 'escape'):
            if self._pipeline:
                self._pipeline.request_interrupt()

    # -- widget update methods (called via call_from_thread) --

    def set_state(self, state: str) -> None:
        self._state = state
        self.query_one(StatusBar).set_state(state)

    def update_meter(self, rms: float, vad_prob: float) -> None:
        self.query_one(AudioMeter).update_level(rms, vad_prob)

    def log_heard(self, text: str) -> None:
        log = self.query_one("#conversation", RichLog)
        log.write(Text.assemble(("  you ", "bold cyan"), (text, "")))

    def log_response(self, text: str) -> None:
        log = self.query_one("#conversation", RichLog)
        log.write(Text.assemble((" evie ", "bold magenta"), (text, "")))

    def log_system(self, text: str) -> None:
        log = self.query_one("#conversation", RichLog)
        log.write(Text(text, style="dim"))

    def log_error(self, text: str) -> None:
        log = self.query_one("#conversation", RichLog)
        log.write(Text(text, style="bold red"))
