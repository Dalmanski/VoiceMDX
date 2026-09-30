import colorsys
import threading
import time
import numpy as np
import soundfile as sf
import customtkinter as ctk

def load_audio(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    return samples, rate

def hsv_hex(h, s, v):
    return "#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h % 1, s, v))

class SpectrumAnalyzer:
    def __init__(self, sample_rate, bars=128, fft_size=4096, min_freq=20, max_freq=16000, db_range=80, attack=0.6, release=0.4, peak_fall=0.03, beat_threshold=0.06):
        self.sample_rate = sample_rate
        self.bars = bars
        self.fft_size = fft_size
        self.db_range = db_range
        self.attack = attack
        self.release = release
        self.peak_fall = peak_fall
        self.beat_threshold = beat_threshold
        self.window = np.hanning(fft_size)
        self.scale = 2 / self.window.sum()
        frequencies = np.fft.rfftfreq(fft_size, 1 / sample_rate)
        edges = np.searchsorted(frequencies, np.geomspace(min_freq, min(sample_rate / 2, max_freq), bars + 1))
        self.low_idx = edges[:-1]
        self.high_idx = np.maximum(edges[1:], self.low_idx + 1)
        self.widths = self.high_idx - self.low_idx
        self.hues = 0.8 * np.arange(bars) / (bars - 1)
        self.hue_offset = 0.0
        self.levels = np.zeros(bars)
        self.peaks = np.zeros(bars)
        self.bass = self.average = self.beat = 0.0

    def reset(self):
        self.levels.fill(0)
        self.peaks.fill(0)
        self.bass = self.average = self.beat = 0.0

    def update(self, audio, position_ms):
        start = max(0, int(position_ms * self.sample_rate / 1000) - self.fft_size // 2)
        chunk = audio[start:start + self.fft_size]
        data = np.pad(chunk, (0, self.fft_size - len(chunk)))
        cumulative = np.concatenate(([0], np.cumsum(np.abs(np.fft.rfft(data * self.window)))))
        band_mean = (cumulative[self.high_idx] - cumulative[self.low_idx]) / self.widths
        target = np.clip((20 * np.log10(band_mean * self.scale + 1e-10) + self.db_range) / self.db_range, 0, 1)
        self.levels += (target - self.levels) * np.where(target > self.levels, self.attack, self.release)
        self.peaks = np.maximum(self.peaks - self.peak_fall, self.levels)
        self.bass = self.levels[:self.bars // 8].mean()
        self.beat = 1.0 if self.bass - self.average > self.beat_threshold and self.beat < 0.5 else self.beat * 0.88
        self.average += (self.bass - self.average) * 0.08
        self.hue_offset = (self.hue_offset + 0.002 + 0.03 * self.bass + 0.03 * self.beat) % 1
        return self.levels

    def colors(self, base, gain):
        return [hsv_hex(self.hue_offset + hue, 1, base + gain * level) for hue, level in zip(self.hues, self.levels)]

class Sparks:
    def __init__(self, count=48):
        self.x = np.zeros(count)
        self.y = np.zeros(count)
        self.vx = np.zeros(count)
        self.vy = np.zeros(count)
        self.life = np.zeros(count)
        self.hue = np.zeros(count)

    def emit(self, xs, ys, hues):
        free = np.flatnonzero(self.life <= 0)[:len(xs)]
        n = len(free)
        self.x[free] = xs[:n]
        self.y[free] = ys[:n]
        self.vx[free] = np.random.uniform(-1.2, 1.2, n)
        self.vy[free] = -np.random.uniform(1.5, 4.5, n)
        self.hue[free] = hues[:n]
        self.life[free] = 1

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.vy *= 0.94
        self.life -= 0.06

class SpectrumWidget:
    panel_width = 68
    top_height = 72
    gradient_steps = 12

    def __init__(self, parent, safe_zone, bottom, bars=64):
        self.parent, self.safe_zone, self.bottom, self.count = parent, safe_zone, bottom, bars
        self.panels, self.canvases, self.visuals = [], [], []
        for _ in range(2):
            panel = ctk.CTkFrame(parent, width=self.panel_width, height=1, corner_radius=0, border_width=0)
            background = panel._apply_appearance_mode(panel.cget("fg_color"))
            canvas = ctk.CTkCanvas(panel, highlightthickness=0, borderwidth=0, bg=background)
            canvas.pack(fill="both", expand=True, padx=5, pady=6)
            panel.pack_propagate(False)
            self.panels.append(panel)
            self.canvases.append(canvas)
            self.visuals.append({"bg": self.shapes(canvas, self.gradient_steps), "glow": self.shapes(canvas, bars), "bars": self.shapes(canvas, bars), "caps": self.shapes(canvas, bars, "#ffffff"), "sparks": self.shapes(canvas, 48)})
        self.top_panel = ctk.CTkFrame(parent, width=1, height=self.top_height, corner_radius=12, border_width=1)
        self.top_base = self.top_panel._apply_appearance_mode(self.top_panel.cget("fg_color"))
        self.top_canvas = ctk.CTkCanvas(self.top_panel, highlightthickness=0, borderwidth=0, bg=self.top_base)
        self.top_canvas.pack(fill="both", expand=True, padx=6, pady=5)
        self.top_panel.pack_propagate(False)
        self.top_bg = self.shapes(self.top_canvas, self.gradient_steps)
        self.top_bars = self.shapes(self.top_canvas, bars)
        self.top_caps = self.shapes(self.top_canvas, bars, "#ffffff")
        self.job = self.audio = self.analyzers = self.loaded = self.sparks = None
        self.started = self.token = 0
        self.visible = False
        for widget in (parent, safe_zone, bottom):
            widget.bind("<Configure>", self.position, add="+")

    def shapes(self, canvas, count, fill=""):
        return [canvas.create_rectangle(0, 0, 0, 0, outline="", fill=fill) for _ in range(count)]

    def position(self, _event=None):
        if not self.visible or not self.safe_zone.winfo_height() or not self.bottom.winfo_height():
            return
        left, width = self.safe_zone.winfo_x(), self.safe_zone.winfo_width()
        parent_width = max(1, self.parent.winfo_width())
        parent_height = max(1, self.parent.winfo_height())
        right = parent_width - left - width
        centers = ((left / 2) / parent_width, (left + width + right / 2) / parent_width)
        y = self.safe_zone.winfo_y() / parent_height
        end = self.bottom.winfo_y() + self.bottom.winfo_height()
        height = max(0, end - self.safe_zone.winfo_y()) / parent_height
        for panel, x in zip(self.panels, centers):
            panel.place(in_=self.parent, relx=x, rely=y, relheight=height, anchor="n")
            panel.lift()
        self.top_panel.place(in_=self.bottom, relx=0.5, rely=0, relwidth=1, y=-10, anchor="s")
        self.top_panel.lift()

    def play(self, path):
        self.stop()
        self.visible = True
        self.safe_zone.grid_configure(pady=(24, self.top_height + 10))
        self.position()
        self.started = time.perf_counter()
        threading.Thread(target=self.load, args=(str(path), self.token), daemon=True).start()
        self.job = self.parent.after(30, self.draw)

    def load(self, path, token):
        try:
            audio, rate = load_audio(path)
        except Exception:
            return
        if token == self.token:
            self.loaded = (token, audio, rate)

    def draw(self):
        if self.loaded:
            token, audio, rate = self.loaded
            self.loaded = None
            if token == self.token:
                right = audio[:, 1] if audio.shape[1] > 1 else audio[:, 0]
                self.audio = (audio[:, 0], right)
                self.analyzers = [SpectrumAnalyzer(rate, bars=self.count) for _ in range(2)]
                self.sparks = [Sparks() for _ in range(2)]
        if self.analyzers:
            self.render()
        self.job = self.parent.after(33, self.draw)

    def gradient(self, canvas, items, analyzer, width, height):
        energy = min(1, analyzer.bass * 1.8 + analyzer.beat * 0.7)
        glow = 0.025 + energy * 0.12
        step = height / len(items)
        for i, item in enumerate(items):
            t = i / max(1, len(items) - 1)
            hue = analyzer.hue_offset + 0.08 * np.sin(t * np.pi)
            value = glow * (0.3 + 0.7 * np.sin(t * np.pi))
            color = hsv_hex(hue, 0.85, value)
            y0, y1 = i * step, (i + 1) * step + 1
            canvas.coords(item, 0, y0, width, y1)
            canvas.itemconfigure(item, fill=color)

    def render(self):
        position = (time.perf_counter() - self.started) * 1000
        for side, (canvas, visual, analyzer, audio) in enumerate(zip(self.canvases, self.visuals, self.analyzers, self.audio)):
            levels = analyzer.update(audio, position)
            width, height = canvas.winfo_width(), canvas.winfo_height()
            self.gradient(canvas, visual["bg"], analyzer, width, height)
            slot, reach = height / self.count, width - 10
            bright, dim = analyzer.colors(0.35, 0.65), analyzer.colors(0.1, 0.3)
            lengths, peaks = np.clip(np.array([levels, analyzer.peaks]) * reach, 1.5, reach)
            for i, (length, peak, color, halo) in enumerate(zip(lengths, peaks, bright, dim)):
                y0 = height - (i + 1) * slot + slot * 0.15
                y1 = y0 + slot * 0.7
                x0, x1 = (width - 4 - length, width - 4) if side == 0 else (4, 4 + length)
                px = width - 4 - peak if side == 0 else 4 + peak
                canvas.coords(visual["glow"][i], x0 - 2, y0, x1 + 2, y1)
                canvas.itemconfigure(visual["glow"][i], fill=halo)
                canvas.coords(visual["bars"][i], x0, y0, x1, y1)
                canvas.itemconfigure(visual["bars"][i], fill=color)
                canvas.coords(visual["caps"][i], px - 1, y0, px + 1, y1)
            if analyzer.beat > 0.99:
                top = np.argsort(levels)[-8:]
                xs = width - 4 - lengths[top] if side == 0 else 4 + lengths[top]
                ys = height - (top + 0.5) * slot
                self.sparks[side].emit(xs, ys, analyzer.hues[top])
            sparks = self.sparks[side]
            sparks.step()
            for item, x, y, life, hue in zip(visual["sparks"], sparks.x, sparks.y, sparks.life, sparks.hue):
                size = 1 + 2 * life
                canvas.coords(item, x - size, y - size, x + size, y + size) if life > 0 else canvas.coords(item, 0, 0, 0, 0)
                if life > 0:
                    canvas.itemconfigure(item, fill=hsv_hex(analyzer.hue_offset + hue, 0.35, life))
        self.render_top()

    def render_top(self):
        levels = np.maximum(self.analyzers[0].levels, self.analyzers[1].levels)
        peaks = np.maximum(self.analyzers[0].peaks, self.analyzers[1].peaks)
        width, height = self.top_canvas.winfo_width(), self.top_canvas.winfo_height()
        self.gradient(self.top_canvas, self.top_bg, self.analyzers[0], width, height)
        slot, reach = width / self.count, height - 4
        colors = self.analyzers[0].colors(0.35, 0.65)
        for i, (level, peak, color) in enumerate(zip(levels, peaks, colors)):
            x0 = i * slot + slot * 0.15
            x1 = x0 + slot * 0.7
            floor = height - 2
            self.top_canvas.coords(self.top_bars[i], x0, floor - level * reach, x1, floor)
            self.top_canvas.itemconfigure(self.top_bars[i], fill=color)
            self.top_canvas.coords(self.top_caps[i], x0, floor - peak * reach - 2, x1, floor - peak * reach - 1)

    def stop(self):
        if self.job:
            self.parent.after_cancel(self.job)
            self.job = None
        self.token += 1
        self.visible = False
        self.audio = self.analyzers = self.loaded = self.sparks = None
        self.safe_zone.grid_configure(pady=(24, 10))
        self.top_panel.place_forget()
        for panel, canvas, visual in zip(self.panels, self.canvases, self.visuals):
            panel.place_forget()
            for items in visual.values():
                for item in items:
                    canvas.coords(item, 0, 0, 0, 0)
        for item in self.top_bg + self.top_bars + self.top_caps:
            self.top_canvas.coords(item, 0, 0, 0, 0)