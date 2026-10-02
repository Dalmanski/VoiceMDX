import asyncio
import os
import site
import sys
import threading
from bisect import bisect_right
from pathlib import Path

import flet as ft
import pygame

MODEL_SIZE = "small"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
LANGUAGE = None
BEAM_SIZE = 5
BEST_OF = 5
MAX_WORDS_PER_LINE = 10
MAX_LINE_SECONDS = 6
MAX_GAP_SECONDS = 1.0
MAX_WORD_SECONDS = 1.5
WORD_HOLD_SECONDS = 0.05
PAST_LINES = 2
FONT_SIZE = 28
WIDTH = 800
INDENT = 48
FILL = 300
COLOR_ACTIVE = "#ffd54a"
COLOR_READ = "#f2f2f2"
COLOR_FUTURE = "#5c6270"
GLOW_ON = ft.BoxShadow(blur_radius=8, color="#18ffd54a")
GLOW_OFF = ft.BoxShadow(blur_radius=8, color="#00ffd54a")
ANIM = ft.Animation(180, ft.AnimationCurve.EASE_OUT)

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
    if words:
        words[0]["start"] = max(words[0]["start"], words[0]["end"] - MAX_WORD_SECONDS)
    for word in words[1:]:
        word["end"] = min(word["end"], word["start"] + MAX_WORD_SECONDS)
    for index in range(len(words) - 1):
        if words[index]["end"] > words[index + 1]["start"]:
            words[index]["end"] = max(words[index]["start"] + 0.05, words[index + 1]["start"])
    return words

def group_words(words):
    lines = []
    current = []

    def flush():
        if current:
            lines.append(list(current))
            current.clear()

    for word in words:
        if current and (word["start"] - current[-1]["end"] > MAX_GAP_SECONDS or len(current) >= MAX_WORDS_PER_LINE or word["end"] - current[0]["start"] >= MAX_LINE_SECONDS):
            flush()
        current.append(word)
        if word["text"].endswith((".", "!", "?", ";", ":")):
            flush()
        elif word["text"].endswith(",") and len(current) >= 4:
            flush()
    flush()
    return lines

def transcribe_file(path):
    whisper = get_model()
    kwargs = {"beam_size": BEAM_SIZE, "best_of": BEST_OF, "vad_filter": False, "condition_on_previous_text": False, "temperature": [0.0, 0.2, 0.4], "word_timestamps": True, "compression_ratio_threshold": 2.4, "log_prob_threshold": -1.0, "no_speech_threshold": 0.7, "hallucination_silence_threshold": 2.0}
    if LANGUAGE:
        kwargs["language"] = LANGUAGE
    segments, _ = whisper.transcribe(str(path), **kwargs)
    lines = []
    last_text = None
    repeats = 0
    for segment in segments:
        words = clean_words(segment)
        if not words:
            continue
        text = " ".join(word["text"] for word in words).lower()
        repeats = repeats + 1 if text == last_text else 0
        last_text = text
        if repeats < 3:
            lines.extend(group_words(words))
    return sorted(lines, key=lambda line: line[0]["start"])

def get_audio_duration(path):
    try:
        from mutagen import File
        audio = File(path)
        return float(audio.info.length) if audio and audio.info else 0.0
    except Exception:
        return 0.0

def format_time(seconds):
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"

class BookReader:
    def __init__(self, page):
        self.page = page
        self.words = []
        self.items = []
        self.starts = []
        self.line_of = []
        self.firsts = []
        self.playing = False
        self.started = False
        self.finished = False
        self.duration = 0.0
        self.generation = 0
        self.revealed = 0
        self.active = -1
        self.line = -1
        self.file_picker = ft.FilePicker()
        self.status = ft.Text("Upload a music or spoken-audio file to begin.", size=16, color="#9aa0aa")
        self.file_name = ft.Text("No file selected", size=14, color="#777d89", expand=True, max_lines=1)
        self.position_text = ft.Text("00:00 / 00:00", size=14, color="#777d89")
        self.progress = ft.ProgressRing(visible=False, width=22, height=22, stroke_width=3)
        self.flow = ft.Row(wrap=True, spacing=0, run_spacing=10, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        self.book = ft.Column(controls=[self.flow, ft.Container(height=400)], scroll=ft.ScrollMode.HIDDEN, horizontal_alignment=ft.CrossAxisAlignment.STRETCH, expand=True)
        self.reader = ft.Container(content=self.message("Upload an audio file to begin."), width=WIDTH, bgcolor="#111217", border_radius=18, padding=28)
        self.upload_button = ft.Button(content="Upload Audio", icon=ft.Icons.UPLOAD_FILE, on_click=self.pick_audio)
        self.play_button = ft.Button(content="Play", icon=ft.Icons.PLAY_ARROW, disabled=True, on_click=self.toggle_play)
        self.reset_button = ft.Button(content="Reset", icon=ft.Icons.DELETE_OUTLINE, disabled=True, on_click=self.reset)
        page.services.append(self.file_picker)
        header = [ft.Text("Book Reader", size=30, weight=ft.FontWeight.BOLD), ft.Text("Transcribe everything first, then read it word by word.", size=14, color="#777d89")]
        body = [ft.Row(controls=[self.upload_button, self.progress], alignment=ft.MainAxisAlignment.CENTER), ft.Row(controls=[self.file_name, self.position_text], width=WIDTH), ft.Row(controls=[self.reader], alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.STRETCH, expand=True)]
        footer = [ft.Row(controls=[self.play_button, self.reset_button], alignment=ft.MainAxisAlignment.CENTER), self.status]
        page.add(ft.SafeArea(content=ft.Column(controls=header + body + footer, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=14, expand=True), expand=True))

    def message(self, text):
        return ft.Column(controls=[ft.Text(text, size=24, color="#777d89")], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)

    def build_book(self, lines):
        self.words = []
        self.line_of = []
        self.firsts = []
        for index, line in enumerate(lines):
            self.firsts.append(len(self.words))
            self.words.extend(line)
            self.line_of.extend([index] * len(line))
        self.starts = [word["start"] for word in self.words]
        self.items = [ft.Container(content=ft.Text(word["text"], size=FONT_SIZE, color=COLOR_FUTURE, weight=ft.FontWeight.W_500), key=f"w{index}", scale=1.0, animate_scale=ANIM, animate=ANIM, border_radius=8, padding=ft.Padding.symmetric(horizontal=4), shadow=GLOW_OFF) for index, word in enumerate(self.words)]
        self.flow.controls = [ft.Container(content=item, padding=ft.Padding.only(left=INDENT)) if index == 0 else item for index, item in enumerate(self.items)] + [ft.Container(width=0) for _ in range(FILL)]
        self.reader.content = self.book

    def paint(self, index):
        item = self.items[index]
        state = 2 if index == self.active else 1 if index < self.revealed else 0
        item.content.color = (COLOR_FUTURE, COLOR_READ, COLOR_ACTIVE)[state]
        item.scale = 1.1 if state == 2 else 1.0
        item.shadow = GLOW_ON if state == 2 else GLOW_OFF

    def apply(self, revealed, active):
        old_revealed, old_active = self.revealed, self.active
        self.revealed, self.active = revealed, active
        changed = set(range(min(old_revealed, revealed), max(old_revealed, revealed))) | {old_active, active}
        for index in changed:
            if 0 <= index < len(self.items):
                self.paint(index)

    def clear(self):
        self.playing = False
        self.started = False
        self.finished = False
        self.generation += 1
        pygame.mixer.music.stop()
        self.words = []
        self.items = []
        self.starts = []
        self.revealed = 0
        self.active = -1
        self.line = -1
        self.duration = 0.0
        self.position_text.value = "00:00 / 00:00"
        self.play_button.content = "Play"
        self.play_button.disabled = True
        self.reset_button.disabled = True

    async def pick_audio(self, e=None):
        files = await self.file_picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=["mp3", "wav", "ogg", "flac", "aac", "m4a", "opus", "webm"])
        if not files:
            return
        file = files[0]
        self.clear()
        self.upload_button.disabled = True
        self.progress.visible = True
        self.file_name.value = file.name
        self.status.value = "Transcribing the complete text..."
        self.reader.content = self.message("Loading...")
        self.page.update()
        try:
            if not file.path:
                raise RuntimeError("Could not access the selected audio file.")
            lines = await asyncio.to_thread(transcribe_file, file.path)
            if not lines:
                raise RuntimeError("No speech was detected in the selected audio.")
            self.build_book(lines)
            self.duration = get_audio_duration(file.path) or self.words[-1]["end"]
            pygame.mixer.music.load(file.path)
            self.play_button.disabled = False
            self.reset_button.disabled = False
            self.status.value = f"Loaded {len(self.words)} words. Press Play."
        except Exception as error:
            self.status.value = f"Error: {type(error).__name__}: {error}"
            self.reader.content = self.message("Something went wrong.")
            print(f"Transcription/playback preparation error: {type(error).__name__}: {error}")
        finally:
            self.upload_button.disabled = False
            self.progress.visible = False
            self.page.update()

    async def toggle_play(self, e=None):
        if not self.words:
            return
        try:
            if self.playing:
                pygame.mixer.music.pause()
                self.playing = False
                self.play_button.content = "Resume"
                self.status.value = "Paused"
            else:
                if self.finished:
                    self.finished = False
                    self.apply(0, -1)
                    self.line = -1
                    pygame.mixer.music.play()
                    await self.book.scroll_to(offset=0, duration=300)
                elif self.started:
                    pygame.mixer.music.unpause()
                else:
                    pygame.mixer.music.play()
                    self.started = True
                self.playing = True
                self.play_button.content = "Pause"
                self.status.value = "Reading..."
                self.generation += 1
                self.page.run_task(self.sync_reader, self.generation)
            self.page.update()
        except Exception as error:
            self.playing = False
            self.status.value = f"Playback error: {type(error).__name__}: {error}"
            print(f"Playback error: {type(error).__name__}: {error}")
            self.page.update()

    async def sync_reader(self, generation):
        try:
            while self.playing and generation == self.generation:
                position = max(0.0, pygame.mixer.music.get_pos() / 1000.0)
                revealed = bisect_right(self.starts, position)
                active = revealed - 1 if revealed and position <= self.words[revealed - 1]["end"] + WORD_HOLD_SECONDS else -1
                self.position_text.value = f"{format_time(position)} / {format_time(self.duration)}"
                if revealed != self.revealed or active != self.active:
                    self.apply(revealed, active)
                    line = self.line_of[max(0, revealed - 1)]
                    if revealed and line != self.line:
                        self.line = line
                        await self.book.scroll_to(scroll_key=f"w{self.firsts[max(0, line - PAST_LINES)]}", duration=400)
                self.page.update()
                if not pygame.mixer.music.get_busy():
                    self.playing = False
                    self.finished = True
                    self.play_button.content = "Replay"
                    self.status.value = "Finished."
                    self.page.update()
                    return
                await asyncio.sleep(0.06)
        except Exception as error:
            self.playing = False
            self.status.value = f"Sync error: {type(error).__name__}: {error}"
            print(f"Sync error: {type(error).__name__}: {error}")
            self.page.update()

    async def reset(self, e=None):
        self.clear()
        self.file_name.value = "No file selected"
        self.reader.content = self.message("Upload an audio file to begin.")
        self.status.value = "Upload a music or spoken-audio file to begin."
        self.page.update()

def main(page: ft.Page):
    page.title = "Book Reader"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0b0c10"
    page.padding = 24
    BookReader(page)

if __name__ == "__main__":
    pygame.init()
    pygame.mixer.init()
    ft.run(main)