import asyncio
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import threading
from bisect import bisect_right
from pathlib import Path

import flet as ft
import pygame

DEBUG = True
MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
LANGUAGE = "en"
BEAM_SIZE = 5
BEST_OF = 5
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
COMPRESSION_LIMIT = 2.4
LOG_PROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.7
MAX_LINE_CHARS = 26
MAX_LINE_SECONDS = 6
MAX_GAP_SECONDS = 1.0
LINES_PER_SCREEN = 2
WIDTH = 900
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
LEAD_SECONDS = 1.0
HOLD_SECONDS = 0.6
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
BACKGROUND_EXTENSIONS = ["mp4", "mov", "mkv", "webm", "avi", "png", "jpg", "jpeg", "webp", "bmp"]
ORIGINAL = {"font": "Arial Black", "weight": ft.FontWeight.W_900, "size": 40, "unsung": "#ffffff", "sung": "#ffe600", "outline_color": "#000000", "outline": 7, "shadow_color": "#000000", "shadow": None, "bg_top": "#1030c8", "bg_bottom": "#050f5e", "video_size": 60, "video_outline": 5, "video_shadow": 0, "margin": 60, "dim": 0}
LIMBUS = {"font": "Pretendard", "weight": ft.FontWeight.W_500, "size": 34, "unsung": "#7b88c9", "sung": "#b9c6ff", "outline_color": "#0a0e2a", "outline": 3, "shadow_color": "#05071a", "shadow": (0.04, 0.08), "bg_top": "#3a3f4d", "bg_bottom": "#0e1016", "video_size": 44, "video_outline": 2, "video_shadow": 4, "margin": 90, "dim": -0.12}
STYLES = {"Original": ORIGINAL, "Limbus Company Karaoke": LIMBUS}

_model = None
_model_lock = threading.Lock()

def debug(message):
    if DEBUG:
        print(f"[DEBUG] {message}", flush=True)

def debug_lines(lines):
    for index, line in enumerate(lines):
        debug(f"line {index:03d} [{line[0]['start']:7.2f} - {line[-1]['end']:7.2f}] {' '.join(word['text'] for word in line)}")

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
                debug(f"model loaded: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
            except Exception as error:
                debug(f"GPU load failed ({type(error).__name__}: {error}), falling back to CPU")
                _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model

def clean_words(segment):
    words = []
    for word in segment.words or []:
        text = word.word.strip()
        if not text or word.start is None or word.end is None:
            continue
        start = max(0.0, float(word.start))
        end = max(start + 0.05, float(word.end))
        words.append({"start": start, "end": end, "text": text})
    return words

def word_key(text):
    return re.sub(r"\W+", "", text.casefold())

def normalize_words(words):
    normalized = []
    for word in sorted(words, key=lambda item: (item["start"], item["end"])):
        if normalized and word_key(word["text"]) == word_key(normalized[-1]["text"]) and word["start"] < normalized[-1]["end"]:
            normalized[-1]["end"] = max(normalized[-1]["end"], word["end"])
            continue
        if normalized and word["start"] < normalized[-1]["end"]:
            normalized[-1]["end"] = max(normalized[-1]["start"] + 0.05, word["start"])
        normalized.append(word)
    return normalized

def group_words(words):
    lines = []
    current = []

    def flush():
        if current:
            lines.append(list(current))
            current.clear()

    for word in words:
        chars = sum(len(item["text"]) for item in current) + len(current) + len(word["text"])
        if current and (word["start"] - current[-1]["end"] > MAX_GAP_SECONDS or chars > MAX_LINE_CHARS or word["end"] - current[0]["start"] >= MAX_LINE_SECONDS):
            flush()
        current.append(word)
        if word["text"].endswith((".", "!", "?", ";", ":")):
            flush()
        elif word["text"].endswith(",") and len(current) >= 4:
            flush()
    flush()
    return lines

def transcribe_file(path):
    from faster_whisper.audio import decode_audio
    whisper = get_model()
    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    total = len(audio) / SAMPLE_RATE
    debug(f"file: {path}")
    debug(f"decoded {len(audio)} samples = {total:.1f}s")
    kwargs = {"language": LANGUAGE, "task": "transcribe", "beam_size": BEAM_SIZE, "best_of": BEST_OF, "vad_filter": False, "condition_on_previous_text": False, "temperature": (0.0, 0.2, 0.4), "word_timestamps": True, "chunk_length": CHUNK_LENGTH, "compression_ratio_threshold": COMPRESSION_LIMIT, "log_prob_threshold": LOG_PROB_THRESHOLD, "no_speech_threshold": NO_SPEECH_THRESHOLD}
    segments, info = whisper.transcribe(audio, **kwargs)
    words = []
    count = 0
    for segment in segments:
        count += 1
        debug(f"segment {segment.start:7.2f}-{segment.end:7.2f} logprob={segment.avg_logprob:.2f} no_speech={segment.no_speech_prob:.2f} compression={segment.compression_ratio:.2f} text={segment.text.strip()[:80]!r}")
        words.extend(clean_words(segment))
    words = normalize_words(words)
    debug(f"language={info.language} probability={info.language_probability:.3f}")
    debug(f"segments={count}")
    debug(f"total words: {len(words)}")
    if words:
        debug(f"first word at {words[0]['start']:.2f}s, last word ends at {words[-1]['end']:.2f}s, audio length {total:.2f}s")
    lines = group_words(words)
    debug(f"total lines: {len(lines)}")
    debug_lines(lines)
    return lines

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

def ass_color(value):
    return f"&H00{value[5:7]}{value[3:5]}{value[1:3]}".upper()

def ass_time(seconds):
    total = max(0, round(seconds * 100))
    return f"{total // 360000}:{total // 6000 % 60:02d}:{total // 100 % 60:02d}.{total % 100:02d}"

def build_ass(lines, style):
    style_line = f"Style: Default,{style['font']},{style['video_size']},{ass_color(style['sung'])},{ass_color(style['unsung'])},{ass_color(style['outline_color'])},{ass_color(style['shadow_color'])},0,0,0,0,100,100,0,0,1,{style['video_outline']},{style['video_shadow']},2,40,40,{style['margin']},1"
    output = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {VIDEO_WIDTH}", f"PlayResY: {VIDEO_HEIGHT}", "WrapStyle: 2", "", "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding", style_line, "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    blocks = [lines[index:index + LINES_PER_SCREEN] for index in range(0, len(lines), LINES_PER_SCREEN)]
    previous_end = 0.0
    for number, block in enumerate(blocks):
        words = [word for line in block for word in line]
        start = max(words[0]["start"] - LEAD_SECONDS, previous_end)
        end = words[-1]["end"] + HOLD_SECONDS
        if number + 1 < len(blocks):
            end = min(end, blocks[number + 1][0][0]["start"])
        previous_end = end
        starts = [word["start"] for word in words] + [words[-1]["end"]]
        count = 0
        rows = []
        for line in block:
            row = ""
            for word in line:
                row += f"{{\\k{round((starts[count + 1] - starts[count]) * 100)}}}{word['text']} "
                count += 1
            rows.append(row.strip())
        lead = f"{{\\k{round((words[0]['start'] - start) * 100)}}}"
        output.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{lead}" + "\\N".join(rows))
    debug(f"subtitle blocks written: {len(blocks)}")
    return "\n".join(output)

def make_background(path, style):
    surface = pygame.Surface((VIDEO_WIDTH, VIDEO_HEIGHT))
    top = pygame.Color(style["bg_top"])
    bottom = pygame.Color(style["bg_bottom"])
    for y in range(VIDEO_HEIGHT):
        pygame.draw.line(surface, top.lerp(bottom, y / (VIDEO_HEIGHT - 1)), (0, y), (VIDEO_WIDTH, y))
    pygame.image.save(surface, str(path))

def render_video(audio, background, output, lines, style):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg not found. Install it and add it to PATH.")
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, "lyrics.ass").write_text(build_ass(lines, style), encoding="utf-8")
        background = Path(background).resolve() if background else Path(folder, "bg.png")
        if not background.exists():
            make_background(background, style)
        source = ["-loop", "1", "-framerate", "25", "-i", str(background)] if background.suffix.lower() in IMAGE_EXTENSIONS else ["-stream_loop", "-1", "-i", str(background)]
        fit = f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},eq=brightness={style['dim']},ass=lyrics.ass"
        command = ["ffmpeg", "-y"] + source + ["-i", str(Path(audio).resolve()), "-vf", fit, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", "-shortest", str(Path(output).resolve())]
        debug(f"ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
        if result.returncode:
            debug(f"ffmpeg stderr tail: {result.stderr[-800:]}")
            raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "FFmpeg failed.")

def make_word(text, style):
    def glyph(color, stroke=0):
        options = {"size": style["size"], "font_family": style["font"], "weight": style["weight"]}
        if stroke:
            return ft.Text(spans=[ft.TextSpan(text, style=ft.TextStyle(foreground=ft.Paint(color=color, stroke_width=stroke, style=ft.PaintingStyle.STROKE), **options))])
        return ft.Text(text, color=color, **options)

    layers = []
    if style["shadow"]:
        layers.append(ft.Container(content=glyph(style["shadow_color"]), offset=ft.Offset(*style["shadow"]), opacity=0.8))
    layers.append(glyph(style["outline_color"], style["outline"]))
    layers.append(glyph(style["unsung"]))
    return ft.Container(content=ft.Stack(controls=layers), padding=ft.Padding.symmetric(horizontal=7))

class BookReader:
    def __init__(self, page):
        self.page = page
        self.style = ORIGINAL
        self.lines = []
        self.audio_path = ""
        self.background_path = ""
        self.words = []
        self.items = []
        self.rows = []
        self.starts = []
        self.line_of = []
        self.playing = False
        self.started = False
        self.finished = False
        self.duration = 0.0
        self.generation = 0
        self.revealed = 0
        self.block = -1
        self.file_picker = ft.FilePicker()
        self.status = ft.Text("Upload a music or spoken-audio file to begin.", size=16, color="#9aa0aa")
        self.file_name = ft.Text("No file selected", size=14, color="#777d89", expand=True, max_lines=1)
        self.position_text = ft.Text("00:00 / 00:00", size=14, color="#777d89")
        self.progress = ft.ProgressRing(visible=False, width=22, height=22, stroke_width=3)
        self.view = ft.Column(alignment=ft.MainAxisAlignment.END, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=28, expand=True)
        self.reader = ft.Container(content=self.message("Upload an audio file to begin."), width=WIDTH, border_radius=18, padding=28)
        self.refresh_reader()
        self.upload_button = ft.Button(content="Upload Audio", icon=ft.Icons.UPLOAD_FILE, on_click=self.pick_audio)
        self.background_button = ft.Button(content="Background Image/Video", icon=ft.Icons.IMAGE, on_click=self.pick_background)
        self.style_dropdown = ft.Dropdown(value="Original", width=240, options=[ft.DropdownOption(key=name, text=name) for name in STYLES], on_select=self.change_style)
        self.play_button = ft.Button(content="Play", icon=ft.Icons.PLAY_ARROW, disabled=True, on_click=self.toggle_play)
        self.export_button = ft.Button(content="Export Video", icon=ft.Icons.MOVIE, disabled=True, on_click=self.export_video)
        self.reset_button = ft.Button(content="Reset", icon=ft.Icons.DELETE_OUTLINE, disabled=True, on_click=self.reset)
        page.services.append(self.file_picker)
        header = [ft.Text("Karaoke Reader", size=30, weight=ft.FontWeight.BOLD), ft.Text("Transcribe everything first, then sing along word by word.", size=14, color="#777d89")]
        body = [ft.Row(controls=[self.upload_button, self.background_button, self.style_dropdown, self.progress], alignment=ft.MainAxisAlignment.CENTER), ft.Row(controls=[self.file_name, self.position_text], width=WIDTH), ft.Row(controls=[self.reader], alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.STRETCH, expand=True)]
        footer = [ft.Row(controls=[self.play_button, self.export_button, self.reset_button], alignment=ft.MainAxisAlignment.CENTER), self.status]
        page.add(ft.SafeArea(content=ft.Column(controls=header + body + footer, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=14, expand=True), expand=True))

    def message(self, text):
        return ft.Column(controls=[ft.Text(text, size=24, color="#c9d4ff")], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)

    def refresh_reader(self):
        self.reader.gradient = ft.LinearGradient(begin=ft.Alignment(0, -1), end=ft.Alignment(0, 1), colors=[self.style["bg_top"], self.style["bg_bottom"]])
        is_image = self.background_path.lower().endswith(IMAGE_EXTENSIONS)
        self.reader.image = ft.DecorationImage(src=self.background_path, fit=ft.BoxFit.COVER) if is_image else None

    def build_book(self, lines, block=0, revealed=0):
        self.lines = lines
        self.words = [word for line in lines for word in line]
        self.line_of = [index for index, line in enumerate(lines) for _ in line]
        self.starts = [word["start"] for word in self.words]
        self.items = [make_word(word["text"], self.style) for word in self.words]
        self.rows = []
        position = 0
        for line in lines:
            self.rows.append(ft.Row(controls=self.items[position:position + len(line)], alignment=ft.MainAxisAlignment.CENTER, wrap=False, spacing=0))
            position += len(line)
        debug(f"book built: {len(self.words)} words, {len(self.rows)} rows")
        self.revealed = 0
        self.show(block)
        self.apply(revealed)
        self.reader.content = self.view

    def show(self, block):
        self.block = block
        self.view.controls = self.rows[block * LINES_PER_SCREEN:(block + 1) * LINES_PER_SCREEN]

    def apply(self, revealed):
        old = self.revealed
        self.revealed = revealed
        for index in range(min(old, revealed), max(old, revealed)):
            self.items[index].content.controls[-1].color = self.style["sung"] if index < revealed else self.style["unsung"]

    def clear(self):
        self.playing = False
        self.started = False
        self.finished = False
        self.generation += 1
        pygame.mixer.music.stop()
        self.lines = []
        self.audio_path = ""
        self.words = []
        self.items = []
        self.rows = []
        self.starts = []
        self.revealed = 0
        self.block = -1
        self.duration = 0.0
        self.position_text.value = "00:00 / 00:00"
        self.play_button.content = "Play"
        self.play_button.disabled = True
        self.export_button.disabled = True
        self.reset_button.disabled = True

    def change_style(self, e=None):
        self.style = STYLES[self.style_dropdown.value]
        self.refresh_reader()
        if self.lines:
            self.build_book(self.lines, max(self.block, 0), self.revealed)
        debug(f"style changed: {self.style_dropdown.value}")
        self.page.update()

    async def pick_background(self, e=None):
        files = await self.file_picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=BACKGROUND_EXTENSIONS)
        if files and files[0].path:
            self.background_path = files[0].path
            self.refresh_reader()
            debug(f"background: {self.background_path}")
            self.status.value = f"Background: {files[0].name}"
            self.page.update()

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
            self.audio_path = file.path
            self.duration = get_audio_duration(file.path) or self.words[-1]["end"]
            debug(f"duration used for display: {self.duration:.2f}s")
            pygame.mixer.music.load(file.path)
            self.play_button.disabled = False
            self.export_button.disabled = False
            self.reset_button.disabled = False
            self.status.value = f"Loaded {len(self.words)} words. Press Play or Export Video."
        except Exception as error:
            self.status.value = f"Error: {type(error).__name__}: {error}"
            self.reader.content = self.message("Something went wrong.")
            print(f"Transcription/playback preparation error: {type(error).__name__}: {error}")
        finally:
            self.upload_button.disabled = False
            self.progress.visible = False
            self.page.update()

    async def export_video(self, e=None):
        source = Path(self.audio_path)
        output = source.with_name(f"{source.stem}_karaoke.mp4")
        self.export_button.disabled = True
        self.progress.visible = True
        self.status.value = "Rendering video..."
        self.page.update()
        try:
            await asyncio.to_thread(render_video, source, self.background_path, output, self.lines, self.style)
            debug(f"video saved: {output}")
            self.status.value = f"Saved: {output}"
        except Exception as error:
            self.status.value = f"Export error: {type(error).__name__}: {error}"
            print(f"Export error: {type(error).__name__}: {error}")
        finally:
            self.export_button.disabled = False
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
                debug("paused")
            else:
                if self.finished:
                    self.finished = False
                    self.apply(0)
                    self.show(0)
                    pygame.mixer.music.play()
                elif self.started:
                    pygame.mixer.music.unpause()
                else:
                    pygame.mixer.music.play()
                    self.started = True
                self.playing = True
                self.play_button.content = "Pause"
                self.status.value = "Singing..."
                self.generation += 1
                debug(f"playing, generation={self.generation}")
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
                self.position_text.value = f"{format_time(position)} / {format_time(self.duration)}"
                if revealed != self.revealed:
                    self.apply(revealed)
                    block = self.line_of[max(0, revealed - 1)] // LINES_PER_SCREEN
                    if block != self.block:
                        debug(f"position={position:.2f}s revealed={revealed}/{len(self.words)} block {self.block}->{block} word={self.words[max(0, revealed - 1)]['text']!r}")
                        self.show(block)
                self.page.update()
                if not pygame.mixer.music.get_busy():
                    self.playing = False
                    self.finished = True
                    self.apply(len(self.words))
                    self.play_button.content = "Replay"
                    self.status.value = "Finished."
                    debug(f"finished at position={position:.2f}s revealed={self.revealed}/{len(self.words)}")
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
        self.background_path = ""
        self.refresh_reader()
        self.file_name.value = "No file selected"
        self.reader.content = self.message("Upload an audio file to begin.")
        self.status.value = "Upload a music or spoken-audio file to begin."
        self.page.update()

def main(page: ft.Page):
    page.title = "Karaoke Reader"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0b0c10"
    page.padding = 24
    BookReader(page)

if __name__ == "__main__":
    pygame.init()
    pygame.mixer.init()
    ft.run(main)