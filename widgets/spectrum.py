import colorsys
import threading
import time
import numpy as np
import soundfile as sf
import customtkinter as ctk

def load_audio(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    return samples.mean(axis=1), rate

def hsv_hex(h, s, v):
    return "#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h % 1, s, v))

class SpectrumAnalyzer:
    def __init__(self, sample_rate, bars=128, fft_size=8192, min_freq=20, max_freq=16000, db_range=80, attack=0.6, release=0.4, peak_fall=0.03, beat_threshold=0.06):
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

class SpectrumWidget(ctk.CTkFrame):
    def __init__(self, parent, anchor, height=72, bars=64):
        super().__init__(parent, width=1, height=height, corner_radius=12, border_width=1, border_color=("gray55", "gray45"))
        self.anchor = anchor
        self.visual_height = height
        self.count = bars
        self.base = self._apply_appearance_mode(self.cget("fg_color"))
        self.canvas = ctk.CTkCanvas(self, height=height, highlightthickness=0, borderwidth=0, bg=self.base)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=5)
        self.job = self.audio = self.analyzer = self.loaded = self.sparks = None
        self.started = self.token = 0
        self.glow_items = self.shapes(bars)
        self.floor = self.shapes(1)[0]
        self.bars = self.shapes(bars)
        self.caps = self.shapes(bars, "#ffffff")
        self.spark_items = self.shapes(48)
        self.items = self.glow_items + [self.floor] + self.bars + self.caps + self.spark_items

    def shapes(self, count, fill=""):
        return [self.canvas.create_rectangle(0, 0, 0, 0, outline="", fill=fill) for _ in range(count)]

    def play(self, path):
        self.stop()
        self.place(in_=self.anchor, relx=0.5, rely=0, relwidth=1, y=-10, anchor="s")
        self.lift()
        self.started = time.perf_counter()
        threading.Thread(target=self.load, args=(str(path), self.token), daemon=True).start()
        self.job = self.after(30, self.draw)

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
                self.audio = audio
                self.analyzer = SpectrumAnalyzer(rate, bars=self.count)
                self.sparks = Sparks()
        if self.analyzer:
            self.render()
        self.job = self.after(30, self.draw)

    def render(self):
        a = self.analyzer
        levels = a.update(self.audio, (time.perf_counter() - self.started) * 1000)
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        floor = height - 3
        reach = floor - 4
        slot = width / self.count
        spread = slot * 0.35
        lift = 3 + 5 * a.beat
        heights, caps = np.clip(np.array([levels, a.peaks]) * reach * (1 + 0.2 * a.beat), 1.5, reach)
        bright = a.colors(0.35, 0.65)
        dim = a.colors(0.1, 0.3)
        self.canvas.configure(bg=hsv_hex(a.hue_offset, 0.8, 0.04 + 0.1 * a.bass + 0.12 * a.beat))
        self.canvas.coords(self.floor, 0, floor, width, floor + 1)
        self.canvas.itemconfigure(self.floor, fill=hsv_hex(a.hue_offset, 0.6, 0.4 + 0.6 * a.beat))
        for i, (h, p, color, halo) in enumerate(zip(heights, caps, bright, dim)):
            x0 = i * slot + slot * 0.15
            x1 = x0 + slot * 0.7
            self.canvas.coords(self.glow_items[i], x0 - spread, floor - h - lift, x1 + spread, floor)
            self.canvas.itemconfigure(self.glow_items[i], fill=halo)
            self.canvas.coords(self.bars[i], x0, floor - h, x1, floor)
            self.canvas.itemconfigure(self.bars[i], fill=color)
            self.canvas.coords(self.caps[i], x0, floor - p - 3, x1, floor - p - 1)
        if a.beat > 0.99:
            top = np.argsort(levels)[-8:]
            self.sparks.emit(top * slot + slot / 2, floor - heights[top], a.hues[top])
        self.sparks.step()
        for item, x, y, life, hue in zip(self.spark_items, self.sparks.x, self.sparks.y, self.sparks.life, self.sparks.hue):
            if life > 0:
                size = 1 + 2 * life
                self.canvas.coords(item, x - size, y - size, x + size, y + size)
                self.canvas.itemconfigure(item, fill=hsv_hex(a.hue_offset + hue, 0.35, life))
            else:
                self.canvas.coords(item, 0, 0, 0, 0)

    def stop(self):
        if self.job:
            self.after_cancel(self.job)
            self.job = None
        self.token += 1
        self.audio = self.analyzer = self.loaded = self.sparks = None
        self.canvas.configure(bg=self.base)
        for item in self.items:
            self.canvas.coords(item, 0, 0, 0, 0)
        self.place_forget()