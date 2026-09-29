import math
import os
import queue
import site
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
import customtkinter as ctk
MODEL_SIZE = "small"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
LANGUAGE = None
BEAM_SIZE = 5
BEST_OF = 5
MAX_WORDS_PER_LINE = 10
MAX_LINE_SECONDS = 6
MAX_GAP_SECONDS = 1.0
INSTRUMENTAL_GAP_SECONDS = 2.5
WORD_LEAD_SECONDS = 0.05
COLOR_PAST = "#ffffff"
COLOR_ACTIVE = "#ffd54a"
COLOR_FUTURE = "#7d7d8c"
COLOR_GLOW = "#ff9d00"
BASE_PX = 26
ACTIVE_PX = 42
IDLE_PX = 34
ROW_H = ACTIVE_PX * 1.3
SPRING_K = 380.0
SPRING_C = 20.0
LINE_FADE_SECONDS = 0.30
LINE_SLIDE_PX = 14
FRAME_MS = 16
GLOW_LAYERS = [(10, 0.08), (7.5, 0.14), (5, 0.24), (3, 0.40), (1.5, 0.60)]
RING = [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]
_model = None
_model_lock = threading.Lock()

def configure_cuda():
    roots = [Path(sys.prefix) / "Lib" / "site-packages"]
    try:
        roots.extend(Path(base) for base in site.getsitepackages())
    except Exception:
        pass
    paths = [root / "nvidia" / lib / "bin" for root in roots for lib in ("cublas", "cudnn")]
    if os.environ.get("CUDA_PATH"):
        paths.append(Path(os.environ["CUDA_PATH"]) / "bin")
    for path in paths:
        if path.is_dir():
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(path))
            except (AttributeError, OSError):
                pass

def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            configure_cuda()
            try:
                _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            except Exception:
                _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model

def clean_words(segment):
    words = []
    for word in segment.words or []:
        text = word.word.strip()
        if not text:
            continue
        start = float(word.start)
        end = float(word.end)
        if end <= start:
            end = start + 0.08
        words.append({"start": start, "end": end, "text": text})
    for index in range(len(words) - 1):
        if words[index]["end"] > words[index + 1]["start"]:
            words[index]["end"] = max(words[index]["start"] + 0.05, words[index + 1]["start"])
    return words

def group_words(words):
    lines = []
    current = []
    def flush():
        if current:
            lines.append({"start": current[0]["start"], "end": current[-1]["end"], "words": list(current)})
            current.clear()
    for word in words:
        if current and (word["start"] - current[-1]["end"] > MAX_GAP_SECONDS or len(current) >= MAX_WORDS_PER_LINE or word["end"] - current[0]["start"] >= MAX_LINE_SECONDS):
            flush()
        current.append(word)
        text = word["text"]
        if text.endswith((".", "!", "?", ";", ":")):
            flush()
        elif text.endswith(",") and len(current) >= 4:
            flush()
    flush()
    return lines

def _transcribe_worker(path, generation, stop_event, events):
    try:
        whisper = get_model()
        kwargs = {"beam_size": BEAM_SIZE, "best_of": BEST_OF, "vad_filter": False, "condition_on_previous_text": False, "temperature": [0.0, 0.2, 0.4], "word_timestamps": True, "compression_ratio_threshold": 2.4, "log_prob_threshold": -1.0, "no_speech_threshold": 0.7, "hallucination_silence_threshold": 2.0}
        if LANGUAGE:
            kwargs["language"] = LANGUAGE
        segments, _ = whisper.transcribe(str(path), **kwargs)
        last_text = None
        repeats = 0
        for segment in segments:
            if stop_event.is_set():
                return
            words = clean_words(segment)
            if not words:
                continue
            text = " ".join(word["text"] for word in words).lower()
            repeats = repeats + 1 if text == last_text else 0
            last_text = text
            if repeats >= 3:
                continue
            events.put((generation, "lines", group_words(words)))
    except Exception as error:
        events.put((generation, "error", f"Lyrics transcription failed: {type(error).__name__}: {error}"))

def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)

def _hex(color):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v)))) for v in color)

class LyricsOverlay:
    def __init__(self, parent, spectrum, on_error=None):
        self.parent = parent
        self.spectrum = spectrum
        self.on_error = on_error
        self.events = queue.Queue()
        self.lines = []
        self.generation = 0
        self.stop_event = None
        self.started = 0
        self.visible = False
        self.poll_job = None
        self.last_frame = 0
        self.fonts = {}
        self.pools = {"text": [], "line": []}
        self.cursors = dict.fromkeys(self.pools, 0)
        self.cur_line = None
        self.line_t0 = 0
        self.words = []
        self.rows = []
        self.n_rows = 1
        self.box_w = 150
        self.box_h = 76
        self.container = ctk.CTkFrame(spectrum.top_panel, width=180, height=76, corner_radius=12, border_width=1, border_color=spectrum.top_panel.cget("border_color"), fg_color=spectrum.top_panel.cget("fg_color"))
        self.canvas = tk.Canvas(self.container, background=spectrum.top_base, highlightthickness=0, borderwidth=0, takefocus=0, cursor="arrow")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=5)
        self.bg = self._rgb(spectrum.top_base)
        self.c_past = self._rgb(COLOR_PAST)
        self.c_active = self._rgb(COLOR_ACTIVE)
        self.c_future = self._rgb(COLOR_FUTURE)
        self.c_glow = self._rgb(COLOR_GLOW)
        spectrum.top_panel.bind("<Configure>", self.position, add="+")

    def _rgb(self, color):
        return tuple(v / 257.0 for v in self.canvas.winfo_rgb(color))

    def _font(self, px):
        px = max(8, int(round(px)))
        if px not in self.fonts:
            self.fonts[px] = tkfont.Font(root=self.parent, family="Segoe UI", size=-px, weight="bold")
        return self.fonts[px]

    def position(self, _event=None):
        if not self.visible or not self.spectrum.top_panel.winfo_height():
            return
        panel_width = max(1, self.spectrum.top_panel.winfo_width())
        panel_height = max(1, self.spectrum.top_panel.winfo_height())
        width = min(max(140, panel_width - 24), self.box_w)
        self.container.place(relx=0.5, rely=0.5, relwidth=min(1, width / panel_width), relheight=min(1, self.box_h / panel_height), anchor="center")
        self.container.lift()

    def start(self, path):
        self.stop()
        self.generation += 1
        self.started = time.perf_counter()
        self.last_frame = self.started
        self.stop_event = threading.Event()
        self.visible = True
        self._prepare_idle()
        threading.Thread(target=_transcribe_worker, args=(path, self.generation, self.stop_event, self.events), daemon=True).start()
        self._tick()

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
            self.stop_event = None
        if self.poll_job is not None:
            self.parent.after_cancel(self.poll_job)
            self.poll_job = None
        self.generation += 1
        self.visible = False
        self.lines = []
        self.words = []
        self.cur_line = None
        self.canvas.delete("all")
        for pool in self.pools.values():
            pool.clear()
        self.container.place_forget()

    def _prepare_idle(self):
        self.words = []
        self.rows = []
        self.n_rows = 1
        self.box_w = 150
        self.box_h = 76
        self.position()

    def _prepare_line(self, line_index):
        self.words = [{"text": w["text"], "start": w["start"], "end": w["end"], "a": 0.0, "v": 0.0, "col": self.c_future} for w in self.lines[line_index]["words"]]
        panel_width = self.spectrum.top_panel.winfo_width()
        max_width = max(220, panel_width - 100) if panel_width > 1 else 700
        base = self._font(BASE_PX)
        space = base.measure(" ") + 4
        rows = [[]]
        row_width = 0
        widest = 0
        for index, word in enumerate(self.words):
            width = base.measure(word["text"])
            if rows[-1] and row_width + space + width > max_width:
                rows.append([])
                row_width = 0
            row_width += (space if rows[-1] else 0) + width
            widest = max(widest, row_width)
            rows[-1].append(index)
        self.rows = rows
        self.n_rows = len(rows)
        self.box_w = int(widest + 130)
        self.box_h = int(self.n_rows * ROW_H + 22)
        self.position()

    def _drain_events(self):
        while True:
            try:
                generation, kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if generation != self.generation:
                continue
            if kind == "lines":
                self.lines.extend(payload)
            elif kind == "error" and self.on_error:
                self.on_error(payload)

    def _locate(self, now):
        line_index = -1
        for index, line in enumerate(self.lines):
            if line["start"] > now:
                break
            line_index = index
        word_index = -1
        if line_index >= 0:
            line = self.lines[line_index]
            next_start = self.lines[line_index + 1]["start"] if line_index + 1 < len(self.lines) else None
            if now > line["end"] + INSTRUMENTAL_GAP_SECONDS and (next_start is None or now < next_start):
                line_index = -1
            else:
                for index, word in enumerate(line["words"]):
                    if word["start"] > now:
                        break
                    word_index = index
        return line_index, word_index

    def _tick(self):
        if not self.visible:
            return
        self._drain_events()
        real = time.perf_counter()
        dt = min(0.05, max(0.001, real - self.last_frame))
        self.last_frame = real
        now = real - self.started + WORD_LEAD_SECONDS
        line_index, word_index = self._locate(now)
        if line_index != self.cur_line:
            self.cur_line = line_index
            self.line_t0 = real
            if line_index >= 0:
                self._prepare_line(line_index)
            else:
                self._prepare_idle()
        self._draw(real, now, dt, word_index)
        self.poll_job = self.parent.after(FRAME_MS, self._tick)

    def _put(self, kind, coords, **options):
        pool = self.pools[kind]
        index = self.cursors[kind]
        if index == len(pool):
            pool.append(getattr(self.canvas, "create_" + kind)(*coords, **options))
        else:
            self.canvas.coords(pool[index], *coords)
            self.canvas.itemconfigure(pool[index], state="normal", **options)
        self.cursors[kind] += 1

    def _text(self, x, y, text, font, fill, anchor):
        self._put("text", (x, y), text=text, font=font, fill=fill, anchor=anchor)

    def _glow_text(self, x, y, text, font, anchor, strength, scale, fade):
        for radius, alpha in GLOW_LAYERS:
            color = _hex(_lerp(self.bg, self.c_glow, min(1.0, alpha * strength * fade)))
            for dx, dy in RING:
                self._text(x + dx * radius * scale, y + dy * radius * scale, text, font, color, anchor)

    def _draw(self, real, now, dt, word_index):
        self.cursors = dict.fromkeys(self.pools, 0)
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1:
            width = max(1, self.box_w - 16)
        if height <= 1:
            height = max(1, self.box_h - 10)
        if self.cur_line is not None and self.cur_line >= 0 and self.words:
            self._draw_line(real, now, dt, word_index, width, height)
        else:
            self._draw_idle(real, width, height)
        for kind, pool in self.pools.items():
            for item in pool[self.cursors[kind]:]:
                self.canvas.itemconfigure(item, state="hidden")

    def _draw_idle(self, real, width, height):
        pulse = 0.5 + 0.5 * math.sin(real * 4)
        font = self._font(IDLE_PX)
        cx = width / 2
        cy = height / 2 + math.sin(real * 3) * 4
        self._glow_text(cx, cy, "\u266a", font, "center", 0.5 + 0.5 * pulse, 1.0, 1.0)
        self._text(cx, cy, "\u266a", font, _hex(_lerp(self.c_active, self.c_past, pulse * 0.35)), "center")

    def _draw_line(self, real, now, dt, word_index, width, height):
        fade = 1 - (1 - min(1.0, (real - self.line_t0) / LINE_FADE_SECONDS)) ** 3
        scale = min(1.0, (height - 6) / max(1.0, self.n_rows * ROW_H))
        row_h = ROW_H * scale
        top = (height - self.n_rows * row_h) / 2 + (1 - fade) * LINE_SLIDE_PX
        space = self._font(BASE_PX * scale).measure(" ") + 4 * scale
        follow = min(1.0, dt * 12)
        for row_number, row in enumerate(self.rows):
            metas = []
            total = -space
            for index in row:
                word = self.words[index]
                word["v"] += (SPRING_K * ((1.0 if index == word_index else 0.0) - word["a"]) - SPRING_C * word["v"]) * dt
                word["a"] = max(-0.15, min(1.35, word["a"] + word["v"] * dt))
                target_color = self.c_past if index < word_index else self.c_active if index == word_index else self.c_future
                word["col"] = _lerp(word["col"], target_color, follow)
                font = self._font((BASE_PX + (ACTIVE_PX - BASE_PX) * max(0.0, word["a"])) * scale)
                w = font.measure(word["text"])
                metas.append((index, word, font, w))
                total += w + space
            x = (width - total) / 2
            row_bottom = top + (row_number + 1) * row_h
            y = row_bottom - 5 * scale
            for index, word, font, w in metas:
                amount = max(0.0, min(1.0, word["a"]))
                active = index == word_index
                if amount > 0.03:
                    pulse = 0.85 + 0.15 * math.sin(real * 8) if active else 1.0
                    self._glow_text(x, y, word["text"], font, "sw", amount * pulse, scale * (0.8 + 0.4 * amount), fade)
                self._text(x, y, word["text"], font, _hex(_lerp(self.bg, word["col"], fade)), "sw")
                if active:
                    progress = max(0.0, min(1.0, (now - word["start"]) / max(0.05, word["end"] - word["start"])))
                    self._put("line", (x, row_bottom - 1.5 * scale, x + w * progress, row_bottom - 1.5 * scale), fill=_hex(_lerp(self.bg, self.c_glow, fade)), width=2, capstyle="round")
                x += w + space