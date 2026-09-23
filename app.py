import modules.offline as offline
import os
import sys
import gc
import re
import shutil
import tempfile
import threading
import subprocess
import time
import atexit
import importlib.metadata
import importlib.util
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk
import torch
from widgets.console_textbox import ConsoleRedirect, ConsoleTextBox
from utils.ctk_theme import configure_ctk_theme
from utils.vid2wav import convert_media_to_wav

configure_ctk_theme()

uvr_spec = importlib.util.spec_from_file_location("uvr_mdx", Path(__file__).resolve().parent / "utils" / "uvr-mdx.py")
uvr_mdx = importlib.util.module_from_spec(uvr_spec)
uvr_spec.loader.exec_module(uvr_mdx)
seed_spec = importlib.util.spec_from_file_location("seed_vc", Path(__file__).resolve().parent / "utils" / "seed-vc.py")
seed_vc = importlib.util.module_from_spec(seed_spec)
seed_spec.loader.exec_module(seed_vc)

BASE_DIR = Path(__file__).resolve().parent
FFMPEG = BASE_DIR / "ffmpeg.exe"
TEMP_ROOT = Path(tempfile.gettempdir()) / "voicemdx_temp"
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".caf"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".ts", ".mts", ".m2ts"]
ALL_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS
MODES = ["Vocalize", "Speech"]
DIFFUSION_STEP_OPTIONS = {"Low": 25, "Recommended": 50, "High": 75, "Extreme": 100}
FOLLOW_PITCH_OPTIONS = ["Target Voice Pitch", "Source Voice Pitch"]
SAFE_PADDING = 90
CARD_GAP = 16
SOURCE_TARGET_HEIGHT = 118
PROCESS_CARD_HEIGHT = 238
OUTPUT_CARD_HEIGHT = 250
MIN_SEMITONE = -72
MAX_SEMITONE = 72

def version(package):
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None

def cleanup_old_sessions(active=None):
    try:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        for path in TEMP_ROOT.glob("session_*"):
            if not (active and path.resolve() == Path(active).resolve()):
                shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("VoiceMDX - Voice Conversion in Music")
        self.geometry("1240x1080")
        self.minsize(980, 900)
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.mode_var = ctk.StringVar(value="Vocalize")
        self.steps_var = ctk.IntVar(value=50)
        self.steps_choice_var = ctk.StringVar(value="Recommended")
        self.follow_pitch_var = ctk.StringVar(value="Target Voice Pitch")
        self.semitone_var = ctk.IntVar(value=0)
        self.source_path = self.target_path = self.output_path = None
        self.has_background = True
        self.source_wav = self.target_wav = self.seed_source_wav = None
        self.uvr_vocal_path = self.inst_path = self.target_vocal_path = None
        self.converted_path = self.conv_prev = None
        self.src_prev_wav = self.tgt_prev_wav = None
        self.tts_process = None
        self.tts_output_path = None
        self.mic_process = None
        self.mic_output_path = None
        self.process = self.preview_process = self.preview_kind = None
        self.preview_loop = False
        self.generating = self.separating = self.target_loading = False
        self.ffmpeg = str(FFMPEG) if FFMPEG.exists() else None
        self.session_dir = TEMP_ROOT / f"session_{os.getpid()}_{int(time.time())}"
        self.inputs_dir = self.session_dir / "inputs"
        self.preview_dir = self.session_dir / "preview"
        self.inst_dir = self.session_dir / "instrumental"
        self.vocal_dir = self.session_dir / "vocal"
        self.target_vocal_dir = self.session_dir / "target_vocal"
        self.normalized_dir = self.session_dir / "normalized"
        self.seed_output_dir = self.session_dir / "seed_output"
        self.output_dir = self.session_dir / "output"
        cleanup_old_sessions(self.session_dir)
        for path in [self.inputs_dir, self.preview_dir, self.inst_dir, self.vocal_dir, self.target_vocal_dir, self.normalized_dir, self.seed_output_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.build_ui()
        self.original_stdout = sys.stdout
        self.console_stdout = ConsoleRedirect(self.console, self.original_stdout)
        sys.stdout = self.console_stdout
        self.upd_config()
        self.upd_semitone()
        self.after(100, lambda: self.state("zoomed"))
        print(seed_vc.patch_bigvgan())
        print(f"Offline mode: {'enabled' if offline.OFFLINE_MODE else 'disabled'}")
        self.check_env()
        atexit.register(self.cleanup)

    def build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.safe_padding = SAFE_PADDING
        self.card_width = max(480, (self.winfo_screenwidth() - self.safe_padding * 2 - CARD_GAP) // 2)
        safe_zone = ctk.CTkFrame(self, corner_radius=0, border_width=0, fg_color="transparent")
        safe_zone.grid(row=0, column=0, sticky="nsew", padx=self.safe_padding, pady=(24, 10))
        safe_zone.grid_rowconfigure(0, weight=1)
        safe_zone.grid_columnconfigure(0, weight=1)
        cards = ctk.CTkScrollableFrame(safe_zone, corner_radius=0, border_width=0, fg_color="transparent")
        cards.grid(row=0, column=0, sticky="nsew")
        cards.grid_columnconfigure(0, weight=0, minsize=self.card_width)
        cards.grid_columnconfigure(1, weight=0, minsize=self.card_width)
        self.cards_scrollable = cards
        self.source_card = self.file_card(cards, 0, "📌 Source", "Choose the full song, audio, or video.", self.select_source, "source")
        self.source_card["frame"].grid(row=0, column=0, sticky="nsew", pady=(0, 8), padx=(0, 8))
        self.target_card = self.file_card(cards, 0, "📌 Target Voice", "Choose the target voice reference.", self.select_target, "target")
        self.target_card["frame"].grid(row=0, column=1, sticky="nsew", pady=(0, 8), padx=(8, 0))
        divider = ctk.CTkFrame(cards, width=self.card_width * 2 + CARD_GAP, height=2, corner_radius=0, border_width=0, fg_color=("gray75", "gray25"))
        divider.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 8))
        self.stem_card = self.stem_ui(cards, 2, 0)
        self.settings_card = self.settings_ui(cards, 2, 1)
        self.output_card = self.output_ui(cards, 3, 0)
        console_card = ctk.CTkFrame(cards, width=self.card_width, height=OUTPUT_CARD_HEIGHT, corner_radius=12, border_width=0)
        console_card.grid(row=3, column=1, sticky="nsew", pady=7, padx=(8, 0))
        console_card.grid_propagate(False)
        console_card.grid_rowconfigure(0, weight=1)
        console_card.grid_columnconfigure(0, weight=1)
        self.console = ConsoleTextBox(console_card, height=OUTPUT_CARD_HEIGHT - 20, wrap="none", font=ctk.CTkFont(family="Consolas", size=11))
        self.console.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        bottom = ctk.CTkFrame(self, width=self.card_width * 2 + CARD_GAP, height=78, corner_radius=12)
        bottom.grid(row=1, column=0, sticky="ew", padx=self.safe_padding, pady=(0, 14))
        bottom.grid_propagate(False)
        bottom.grid_columnconfigure(0, weight=1)
        self.generate_button = ctk.CTkButton(bottom, text="Generate Converted Vocal + Mix", command=self.gen_thread, height=52, font=ctk.CTkFont(size=16, weight="bold"))
        self.generate_button.grid(row=0, column=0, padx=(12, 8), pady=12, sticky="ew")
        self.bottom_prev = ctk.CTkButton(bottom, text="▶", command=self.toggle_out_prev, width=52, height=52, font=ctk.CTkFont(size=18), state="disabled")
        self.bottom_prev.grid(row=0, column=1, padx=4, pady=12)
        self.bottom_loop = ctk.CTkButton(bottom, text="⟲", command=self.toggle_out_loop, width=52, height=52, font=ctk.CTkFont(size=18), state="disabled")
        self.bottom_loop.grid(row=0, column=2, padx=4, pady=12)
        self.bottom_dl = ctk.CTkButton(bottom, text="⬇", command=self.dl_output, width=46, height=52, font=ctk.CTkFont(size=18), state="disabled")
        self.bottom_dl.grid(row=0, column=3, padx=4, pady=12)
        self.clear_button = ctk.CTkButton(bottom, text="Clear", command=self.clear_console, height=52, width=110)
        self.clear_button.grid(row=0, column=4, padx=(8, 12), pady=12)

    def log(self, text, color=None, live=False):
        self.console.log(text, color=color, live=live)

    def file_card(self, parent, row, title, subtitle, command, kind):
        card = ctk.CTkFrame(parent, width=self.card_width, height=SOURCE_TARGET_HEIGHT, corner_radius=12)
        card.grid(row=row, column=0, sticky="nsew", pady=7)
        card.grid_propagate(False)
        card.grid_columnconfigure(0, weight=0)
        card.grid_columnconfigure(1, weight=1)
        card.grid_columnconfigure(2, weight=0)
        card.grid_columnconfigure(3, weight=0)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=(16, 10), pady=(14, 4), sticky="w")
        name = ctk.CTkLabel(card, text="No file selected", anchor="w", text_color="red")
        name.grid(row=0, column=1, padx=10, pady=(14, 4), sticky="ew")
        actions = ctk.CTkFrame(card, fg_color="transparent", border_width=0)
        actions.grid(row=0, column=2, columnspan=2, padx=(8, 16), pady=(12, 4), sticky="e")
        ctk.CTkButton(actions, text="Choose File", command=command, width=130, height=38).grid(row=0, column=0)
        preview = ctk.CTkButton(actions, text="▶", command=lambda k=kind: self.toggle_prev(k), width=46, height=38, font=ctk.CTkFont(size=18))
        preview.grid(row=0, column=1, padx=(6, 0))
        ctk.CTkLabel(card, text=subtitle, text_color="gray70", anchor="w").grid(row=1, column=0, columnspan=2, padx=16, pady=(0, 10), sticky="w")
        if kind == "source":
            mic_button = ctk.CTkButton(card, text="🎙️", command=self.open_mic, width=56, height=34)
            mic_button.grid(row=1, column=2, padx=(8, 6), pady=(0, 10), sticky="e")
            tts_button = ctk.CTkButton(card, text=" 🗣️ TTS", command=self.open_tts, width=120, height=34)
            tts_button.grid(row=1, column=3, padx=(0, 16), pady=(0, 10), sticky="e")
        return {"frame": card, "name": name, "preview": preview}

    def stem_ui(self, parent, row, column):
        card = ctk.CTkFrame(parent, width=self.card_width, height=PROCESS_CARD_HEIGHT, corner_radius=12)
        card.grid(row=row, column=column, sticky="nsew", pady=7, padx=(0, 8) if column == 0 else (8, 0))
        card.grid_propagate(False)
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="📌 Source Separation", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=(16, 10), pady=(15, 12), sticky="w")
        ctk.CTkLabel(card, text="Inst / BG", font=ctk.CTkFont(size=14, weight="bold")).grid(row=1, column=0, padx=16, pady=12, sticky="w")
        self.inst_name = ctk.CTkLabel(card, text="Not separated yet", anchor="w", text_color="gray60")
        self.inst_name.grid(row=1, column=1, padx=12, pady=12, sticky="ew")
        self.inst_actions = ctk.CTkFrame(card, fg_color="transparent", border_width=0, corner_radius=0)
        self.inst_actions.grid(row=1, column=2, columnspan=2, padx=(0, 16), pady=12, sticky="e")
        self.inst_prev = ctk.CTkButton(self.inst_actions, text="▶", command=self.toggle_inst_prev, width=46, height=36, font=ctk.CTkFont(size=18), state="disabled")
        self.inst_prev.grid(row=0, column=0, padx=(0, 4))
        self.inst_dl = ctk.CTkButton(self.inst_actions, text="⬇", command=self.dl_inst, width=46, height=36, font=ctk.CTkFont(size=18), state="disabled")
        self.inst_dl.grid(row=0, column=1, padx=(4, 0))
        ctk.CTkLabel(card, text="Vocal", font=ctk.CTkFont(size=14, weight="bold")).grid(row=2, column=0, padx=16, pady=(12, 15), sticky="w")
        self.vocal_name = ctk.CTkLabel(card, text="Not separated yet", anchor="w", text_color="gray60")
        self.vocal_name.grid(row=2, column=1, padx=12, pady=(12, 15), sticky="ew")
        self.vocal_actions = ctk.CTkFrame(card, fg_color="transparent", border_width=0, corner_radius=0)
        self.vocal_actions.grid(row=2, column=2, columnspan=2, padx=(0, 16), pady=(12, 15), sticky="e")
        self.vocal_prev = ctk.CTkButton(self.vocal_actions, text="▶", command=self.toggle_vocal_prev, width=46, height=36, font=ctk.CTkFont(size=18), state="disabled")
        self.vocal_prev.grid(row=0, column=0, padx=(0, 4))
        self.vocal_dl = ctk.CTkButton(self.vocal_actions, text="⬇", command=self.dl_vocal, width=46, height=36, font=ctk.CTkFont(size=18), state="disabled")
        self.vocal_dl.grid(row=0, column=1, padx=(4, 0))
        return card

    def settings_ui(self, parent, row, column):
        card = ctk.CTkFrame(parent, width=self.card_width, height=PROCESS_CARD_HEIGHT, corner_radius=12)
        card.grid(row=row, column=column, sticky="nsew", pady=7, padx=(0, 8) if column == 0 else (8, 0))
        card.grid_propagate(False)
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="⚙️ Voice Settings", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=16, pady=(14, 7), sticky="w")
        quality_frame = ctk.CTkFrame(card)
        quality_frame.grid(row=1, column=0, padx=(16, 7), pady=6, sticky="ew")
        quality_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(quality_frame, text="VOICE QUALITY", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, padx=(10, 6), pady=8, sticky="w")
        self.steps_menu = ctk.CTkOptionMenu(quality_frame, variable=self.steps_choice_var, values=list(DIFFUSION_STEP_OPTIONS.keys()), command=self.steps_changed, width=150, height=34)
        self.steps_menu.grid(row=0, column=1, padx=(4, 10), pady=6, sticky="e")
        follow_frame = ctk.CTkFrame(card)
        follow_frame.grid(row=1, column=1, padx=(7, 16), pady=6, sticky="ew")
        follow_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(follow_frame, text="FOLLOW PITCH VOICE", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, padx=(10, 6), pady=8, sticky="w")
        self.follow_pitch_menu = ctk.CTkOptionMenu(follow_frame, variable=self.follow_pitch_var, values=FOLLOW_PITCH_OPTIONS, command=self.follow_pitch_changed, width=175, height=34)
        self.follow_pitch_menu.grid(row=0, column=1, padx=(4, 10), pady=6, sticky="e")
        mode_frame = ctk.CTkFrame(card)
        mode_frame.grid(row=2, column=0, padx=(16, 7), pady=6, sticky="ew")
        mode_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(mode_frame, text="VOICE MODE", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, padx=(10, 6), pady=8, sticky="w")
        self.mode_menu = ctk.CTkOptionMenu(mode_frame, variable=self.mode_var, values=MODES, command=self.mode_changed, width=150, height=34)
        self.mode_menu.grid(row=0, column=1, padx=(4, 10), pady=6, sticky="e")
        pitch_frame = ctk.CTkFrame(card)
        pitch_frame.grid(row=2, column=1, padx=(7, 16), pady=6, sticky="ew")
        pitch_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(pitch_frame, text="SEMITONE", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, padx=(10, 4), pady=6, sticky="w")
        pitch_controls = ctk.CTkFrame(pitch_frame, fg_color="transparent")
        pitch_controls.grid(row=0, column=1, padx=(0, 10), pady=6, sticky="e")
        self.semitone_down = ctk.CTkButton(pitch_controls, text="−", command=lambda: self.adjust_semitone(-1), width=32, height=32, font=ctk.CTkFont(size=18, weight="bold"))
        self.semitone_down.grid(row=0, column=0, padx=(0, 2), pady=0)
        self.semitone_value = ctk.CTkLabel(pitch_controls, text="+0", width=46, anchor="center", font=ctk.CTkFont(size=14, weight="bold"))
        self.semitone_value.grid(row=0, column=1, padx=2, pady=0)
        self.semitone_up = ctk.CTkButton(pitch_controls, text="+", command=lambda: self.adjust_semitone(1), width=32, height=32, font=ctk.CTkFont(size=18, weight="bold"))
        self.semitone_up.grid(row=0, column=2, padx=(2, 0), pady=0)
        self.config_info = ctk.CTkLabel(card, text="", text_color="gray70", anchor="w", justify="left")
        self.config_info.grid(row=3, column=0, columnspan=2, padx=16, pady=(4, 12), sticky="w")
        return card

    def output_ui(self, parent, row, column):
        card = ctk.CTkFrame(parent, width=self.card_width, height=OUTPUT_CARD_HEIGHT, corner_radius=12)
        card.grid(row=row, column=column, sticky="nsew", pady=7, padx=(0, 8))
        card.grid_propagate(False)
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="🚩 Final Output", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=16, pady=(15, 4), sticky="w")
        ctk.CTkLabel(card, text="Normalized Instrumental + converted target vocal", text_color="gray70").grid(row=1, column=0, padx=16, pady=(0, 13), sticky="w")
        self.output_name = ctk.CTkLabel(card, text="Show after generate output", anchor="w", width=250, text_color="orange")
        self.output_name.grid(row=0, column=1, rowspan=2, padx=12, pady=12, sticky="ew")
        self.out_prev = ctk.CTkButton(card, text="▶", command=self.toggle_out_prev, width=46, height=38, font=ctk.CTkFont(size=18), state="disabled")
        self.out_prev.grid(row=0, column=2, rowspan=2, padx=4, pady=12)
        self.out_dl = ctk.CTkButton(card, text="⬇", command=self.dl_output, width=46, height=38, font=ctk.CTkFont(size=18), state="disabled")
        self.out_dl.grid(row=0, column=3, rowspan=2, padx=(4, 16), pady=12)
        ctk.CTkLabel(card, text="Converted Vocal Only", font=ctk.CTkFont(size=14, weight="bold")).grid(row=2, column=0, padx=16, pady=(4, 14), sticky="w")
        self.converted_name = ctk.CTkLabel(card, text="Show after generate output", anchor="w", width=250, text_color="orange")
        self.converted_name.grid(row=2, column=1, padx=12, pady=(4, 14), sticky="ew")
        self.conv_prev = ctk.CTkButton(card, text="▶", command=self.toggle_conv_prev, width=46, height=38, font=ctk.CTkFont(size=18), state="disabled")
        self.conv_prev.grid(row=2, column=2, padx=4, pady=(4, 14))
        self.conv_dl = ctk.CTkButton(card, text="⬇", command=self.dl_conv, width=46, height=38, font=ctk.CTkFont(size=18), state="disabled")
        self.conv_dl.grid(row=2, column=3, padx=(4, 16), pady=(4, 14))
        return card

    def mode_changed(self, choice):
        self.mode_var.set(choice)
        print(f"Mode selected: {choice}")
        self.upd_config()

    def steps_changed(self, choice):
        self.steps_var.set(DIFFUSION_STEP_OPTIONS.get(choice, 50))
        self.upd_config()

    def follow_pitch_changed(self, choice):
        self.follow_pitch_var.set(choice)
        self.upd_config()

    def adjust_semitone(self, delta):
        value = max(MIN_SEMITONE, min(MAX_SEMITONE, int(self.semitone_var.get()) + int(delta)))
        self.semitone_var.set(value)
        self.upd_semitone(value)
        self.upd_config()

    def upd_semitone(self, value=None):
        if value is None:
            value = self.semitone_var.get()
        value = max(MIN_SEMITONE, min(MAX_SEMITONE, int(value)))
        self.semitone_var.set(value)
        self.semitone_value.configure(text=f"{value:+d}")

    def upd_config(self):
        vocalize = self.mode_var.get() == "Vocalize"
        target_pitch = self.follow_pitch_var.get() == "Target Voice Pitch"
        descriptions = {(True, True): "same voice with minimal pitch change", (True, False): "same voice with source pitch", (False, True): "exact voice with minimal source pitch", (False, False): "exact voice without source pitch"}
        description = descriptions[(vocalize, target_pitch)]
        cfg = seed_vc.get_config(self)
        self.config_info.configure(text=f"steps={cfg['steps']} · f0={cfg['f0']} · auto_f0={cfg['auto_f0']}    {description}")

    def pick_file(self, title):
        patterns = " ".join(f"*{ext}" for ext in ALL_EXTENSIONS)
        return filedialog.askopenfilename(title=title, filetypes=[("Audio and Video", patterns), ("All Files", "*.*")])

    def open_mic(self):
        if self.generating or self.separating or self.target_loading:
            return
        recorder_path = BASE_DIR / "mic_record.py"
        if not recorder_path.exists():
            print(f"Mic recorder not found: {self._display_path(recorder_path)}")
            print("Status: Mic recorder not found")
            return
        output_path = self.inputs_dir / "mic_source.wav"
        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass
        try:
            self.mic_process = subprocess.Popen([sys.executable, str(recorder_path), "--apply-on-source", str(output_path)], cwd=str(BASE_DIR))
            self.mic_output_path = output_path
            print("Opening Microphone Recorder...")
            self.after(250, self.check_mic)
        except Exception as exc:
            self.mic_process = None
            self.mic_output_path = None
            print(f"Mic recorder error: {type(exc).__name__}: {exc}")
            print("Status: Unable to open Microphone Recorder")

    def open_tts(self):
        if self.generating or self.separating or self.target_loading:
            return
        editor_path = BASE_DIR / "tts_editor.py"
        if not editor_path.exists():
            print(f"TTS editor not found: {self._display_path(editor_path)}")
            print("Status: TTS editor not found")
            return
        output_path = self.inputs_dir / "tts_source.wav"
        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass
        command = [sys.executable, str(editor_path), "--apply-on-source", str(output_path)]
        try:
            self.tts_process = subprocess.Popen(command, cwd=str(BASE_DIR))
            self.tts_output_path = output_path
            print("Opening TTS Editor...")
            print("Status: TTS Editor opened")
            self.after(250, self.check_tts)
        except Exception as exc:
            self.tts_process = None
            self.tts_output_path = None
            print(f"TTS editor error: {type(exc).__name__}: {exc}")
            print("Status: Unable to open TTS Editor")

    def check_mic(self):
        if self.mic_output_path and self.mic_output_path.exists() and self.mic_output_path.stat().st_size > 0:
            self.apply_source_file(self.mic_output_path, "Microphone")
            self.mic_output_path = None
            self.mic_process = None
            return
        if self.mic_process is not None and self.mic_process.poll() is None:
            self.after(250, self.check_mic)
            return
        self.mic_process = None
        self.mic_output_path = None

    def check_tts(self):
        if self.tts_output_path and self.tts_output_path.exists() and self.tts_output_path.stat().st_size > 0:
            self.apply_source_file(self.tts_output_path, "TTS")
            self.tts_output_path = None
            self.tts_process = None
            return
        if self.tts_process is not None and self.tts_process.poll() is None:
            self.after(250, self.check_tts)
            return
        self.tts_process = None
        self.tts_output_path = None

    def apply_source_file(self, wav_path, source_name):
        path = Path(wav_path)
        if not path.exists() or path.stat().st_size <= 0:
            print(f"Status: {source_name} WAV was not created")
            return
        self.stop_prev()
        self.src_prev_wav = path
        self.source_path = path
        self.clear_prev_proc()
        self.source_card["name"].configure(text=path.name, text_color="green")
        print(f"{source_name} source applied: {self._display_path(path)}")
        print(f"Status: {source_name} WAV loaded as Source")
        self.sep_thread()

    def apply_tts_source(self, wav_path):
        self.apply_source_file(wav_path, "TTS")

    def select_source(self):
        if self.generating or self.separating:
            return
        path = self.pick_file("Choose source audio or video")
        if not path:
            return
        self.stop_prev()
        self.src_prev_wav = None
        self.source_path = Path(path)
        self.source_card["name"].configure(text=self.source_path.name, text_color="green")
        self.clear_prev_proc()
        self.has_background = True
        print(f"Source selected: {self._display_path(self.source_path)}")
        print("Status: Source selected")
        self.sep_thread()

    def select_target(self):
        if self.generating or self.target_loading:
            return
        path = self.pick_file("Choose target voice sample")
        if not path:
            return
        self.stop_prev()
        self.tgt_prev_wav = None
        self.target_vocal_path = None
        self.target_path = Path(path)
        self.target_card["name"].configure(text=self.target_path.name, text_color="green")
        print(f"Target selected: {self._display_path(self.target_path)}")
        print("Status: Target selected")

    def sep_thread(self):
        if self.generating or self.separating:
            return
        if not self.source_path:
            print("Select a source first.")
            print("Status: Select source first")
            return
        threading.Thread(target=self.sep_worker, daemon=True).start()

    def sep_worker(self):
        self.separating = True
        print("Status: Automatically separating source vocal and Inst / BG...")
        try:
            uvr_mdx.ensure_source_stems(self)
            print("Status: UVR separation complete")
        except Exception as exc:
            print(f"UVR ERROR: {type(exc).__name__}: {exc}")
            print("Status: UVR separation failed")
        finally:
            self.separating = False

    def norm_audio(self, input_path, name):
        if not self.ffmpeg:
            raise RuntimeError(f"FFmpeg was not found: {FFMPEG}")
        output = self.normalized_dir / f"{Path(input_path).stem}_{name}.wav"
        command = [self.ffmpeg, "-y", "-i", str(input_path), "-af", "loudnorm=I=-16:LRA=11:TP=-1", "-ar", "44100", "-c:a", "pcm_s16le", str(output)]
        print(f"Normalizing: {Path(input_path).name}")
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(f"Loudness normalization failed for {input_path}.")
        return output

    def ext_audio(self, input_path, output_path, label, channels=1):
        if not self.ffmpeg:
            raise RuntimeError(f"FFmpeg was not found: {FFMPEG}")
        convert_media_to_wav(input_path, output_path, ffmpeg_path=self.ffmpeg, channels=channels, sample_rate=44100, label=label)
        print(f"{label.title()} audio ready: {self._display_path(output_path)}")

    def gen_thread(self):
        if self.generating or self.separating or self.target_loading:
            return
        threading.Thread(target=self.gen, daemon=True).start()

    def gen(self):
        if not self.source_path or not self.target_path:
            print("Source and target are required.")
            print("Status: Select source and target")
            return
        if not self.ffmpeg or not seed_vc.INFERENCE_SCRIPT.exists():
            self.check_env()
            return
        self.generating = True
        self.after(0, lambda: self.generate_button.configure(state="disabled", text="Converting + Mixing..."))
        try:
            self.stop_prev()
            print("Status: Preparing separated source vocal and Inst / BG...")
            uvr_mdx.ensure_source_stems(self)
            seed_input = self.uvr_vocal_path
            print("Status: Cleaning target voice...")
            self.target_wav = self.target_vocal_path = uvr_mdx.run_target_uvr(self)
            seed_vc.prepare_source(self, seed_input)
            cfg = seed_vc.get_config(self)
            print(f"Seed-VC steps: {cfg['steps']}")
            print(f"Seed-VC strength: {cfg['cfg']:.2f}")
            print(f"Seed-VC F0: {cfg['f0']}")
            print(f"Seed-VC Follow Pitch Voice: {self.follow_pitch_var.get()}")
            print(f"Seed-VC Semitone Shift: {cfg['pitch']:+d}")
            print("Status: Running Seed-VC...")
            seed_vc.run(self)
            self.converted_path = self.norm_audio(self.converted_path, "converted_vocal")
            self.soften_conv()
            print("Status: Mixing final output...")
            self.mix_final()
            self.after(0, lambda: self.output_name.configure(text=self.output_path.name, text_color="green"))
            self.after(0, lambda: self.converted_name.configure(text=self.converted_path.name, text_color="green"))
            self.after(0, lambda: self.out_prev.configure(state="normal", text="▶"))
            self.after(0, lambda: self.out_dl.configure(state="normal"))
            self.after(0, lambda: self.conv_prev.configure(state="normal", text="▶"))
            self.after(0, lambda: self.conv_dl.configure(state="normal"))
            self.after(0, lambda: self.bottom_prev.configure(state="normal", text="▶"))
            self.after(0, lambda: self.bottom_loop.configure(state="normal", text="⟲"))
            self.after(0, lambda: self.bottom_dl.configure(state="normal"))
            print("Status: Conversion complete")
            print(f"Final output: {self._display_path(self.output_path)}")
        except Exception as exc:
            print(f"GENERATION ERROR: {type(exc).__name__}: {exc}")
            print("Status: Generation failed")
        finally:
            self.cleanup_gpu()
            self.generating = False
            self.after(0, lambda: self.generate_button.configure(state="normal", text="Generate Converted Vocal + Mix"))

    def soften_conv(self):
        if not self.ffmpeg or not self.converted_path or not Path(self.converted_path).exists():
            raise RuntimeError("Converted vocal was not found for softening.")
        output = self.normalized_dir / "converted_vocal_softened.wav"
        filter_chain = "highpass=f=70,lowpass=f=14500,equalizer=f=6200:t=q:w=1.0:g=-3.0,equalizer=f=9000:t=q:w=1.2:g=-2.0,acompressor=threshold=-18dB:ratio=2:attack=8:release=100,alimiter=limit=-2dB"
        command = [self.ffmpeg, "-y", "-i", str(self.converted_path), "-af", filter_chain, "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(output)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not output.exists():
            self.log_process_output(completed.stdout)
            raise RuntimeError("Converted vocal softening failed.")
        self.converted_path = output
        print(f"Softened converted vocal: {self._display_path(self.converted_path)}")

    def mix_final(self):
        self.output_path = self.output_dir / "final_mix.wav"
        if not self.has_background:
            shutil.copy2(self.converted_path, self.output_path)
            print(f"Final vocal created: {self._display_path(self.output_path)}")
            return
        filter_complex = chr(59).join(["[0:a]aresample=44100[a0]", "[1:a]aresample=44100,volume=5dB[a1]", "[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0:normalize=1[mix]", "[mix]loudnorm=I=-16:LRA=11:TP=-1.5[out]"])
        command = [self.ffmpeg, "-y", "-i", str(self.inst_path), "-i", str(self.converted_path), "-filter_complex", filter_complex, "-map", "[out]", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(self.output_path)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not self.output_path.exists():
            self.log_process_output(completed.stdout)
            raise RuntimeError("Final mix failed.")
        print(f"Final mix created: {self._display_path(self.output_path)}")

    def toggle_src_prev(self):
        self.toggle_prev("source")

    def toggle_tgt_prev(self):
        self.toggle_prev("target")

    def toggle_inst_prev(self):
        self.toggle_prev("instrumental")

    def toggle_vocal_prev(self):
        self.toggle_prev("vocal")

    def toggle_out_prev(self):
        self.toggle_prev("output")

    def toggle_out_loop(self):
        self.toggle_prev("output", loop=True)

    def toggle_conv_prev(self):
        self.toggle_prev("converted_vocal")

    def toggle_prev(self, kind, loop=False):
        if self.preview_kind == kind and self.preview_loop == loop:
            self.stop_prev()
            return
        if kind == "source":
            self.play_prev(self.src_prev_wav or self.prep_prev(self.source_path, "source"), kind, loop=loop)
            return
        if kind == "target":
            if self.target_vocal_path and self.target_vocal_path.exists():
                self.tgt_prev_wav = self.target_vocal_path
                self.play_prev(self.tgt_prev_wav, kind, loop=loop)
                return
            if not self.target_path or not self.target_path.exists():
                print("Preview unavailable: target")
                return
            if self.target_loading:
                return
            self.target_loading = True
            self.after(0, lambda: self.target_card["preview"].configure(state="disabled", text="…"))
            print("Status: Cleaning target for preview...")
            threading.Thread(target=self.prep_tgt_prev, daemon=True).start()
            return
        paths = {"instrumental": self.inst_path, "vocal": self.uvr_vocal_path, "converted_vocal": self.converted_path}
        self.play_prev(paths.get(kind, self.output_path), kind, loop=loop)

    def prep_tgt_prev(self):
        try:
            target = uvr_mdx.run_target_uvr(self)
            self.after(0, lambda: self.target_card["preview"].configure(state="normal", text="▶"))
            print("Status: Target vocal ready")
            self.after(0, lambda p=target: self.play_prev(p, "target"))
        except Exception as exc:
            print(f"TARGET PREVIEW ERROR: {type(exc).__name__}: {exc}")
            print("Status: Target preview failed")
            self.after(0, lambda: self.target_card["preview"].configure(state="normal", text="▶"))
        finally:
            self.target_loading = False

    def play_prev(self, path, kind, loop=False):
        if not path or not Path(path).exists():
            print(f"Preview unavailable: {kind}")
            return
        self.stop_prev()
        try:
            if sys.platform.startswith("win"):
                import winsound
                flags = winsound.SND_FILENAME | winsound.SND_ASYNC
                if loop:
                    flags |= winsound.SND_LOOP
                winsound.PlaySound(str(path), flags)
                self.preview_process = "winsound"
            else:
                ffplay = shutil.which("ffplay")
                if not ffplay:
                    raise RuntimeError("ffplay was not found.")
                command = [ffplay, "-nodisp", "-autoexit"]
                if loop:
                    command += ["-loop", "0"]
                command.append(str(path))
                self.preview_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.preview_kind = kind
            self.preview_loop = loop
            self.set_prev_icons(kind)
            print(f"Playing {kind} preview{' in loop' if loop else ''}: {self._display_path(path)}")
        except Exception as exc:
            self.preview_process = self.preview_kind = None
            self.preview_loop = False
            self.set_prev_icons()
            print(f"Preview error: {type(exc).__name__}: {exc}")

    def prep_prev(self, source_path, kind):
        if not source_path or not Path(source_path).exists() or not self.ffmpeg:
            return None
        output = self.preview_dir / f"{kind}.wav"
        command = [self.ffmpeg, "-y", "-i", str(source_path), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(output)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not output.exists():
            return None
        if kind == "source":
            self.src_prev_wav = output
        return output

    def set_prev_icons(self, active=None):
        buttons = [("source", self.source_card["preview"]), ("target", self.target_card["preview"]), ("instrumental", self.inst_prev), ("vocal", self.vocal_prev), ("output", self.out_prev), ("converted_vocal", self.conv_prev), ("output", self.bottom_prev)]
        for kind, button in buttons:
            button.configure(text="■" if kind == active else "▶")
        if hasattr(self, "bottom_loop"):
            self.bottom_loop.configure(text="■" if self.preview_kind == "output" and self.preview_loop else "⟲")

    def stop_prev(self):
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        if isinstance(self.preview_process, subprocess.Popen):
            try:
                self.preview_process.terminate()
            except Exception:
                pass
        self.preview_process = self.preview_kind = None
        self.preview_loop = False
        if hasattr(self, "source_card"):
            self.set_prev_icons()

    def check_env(self):
        problems = []
        ffmpeg_status = self._display_path(self.ffmpeg) if self.ffmpeg else f"missing: {self._display_path(FFMPEG)}"
        print(f"FFmpeg: {ffmpeg_status}")
        print(f"Seed-VC: {'found' if seed_vc.INFERENCE_SCRIPT.exists() else 'missing'}")
        print(f"Inst_HQ_4: {'found' if uvr_mdx.UVR_INSTRUMENT_MODEL.exists() else 'missing'}")
        print(f"Voc_FT: {'found' if uvr_mdx.UVR_VOCAL_MODEL.exists() else 'missing'}")
        print(f"audio-separator: {'available' if uvr_mdx.Separator is not None else 'missing'}")
        print(f"audio-separator version: {version('audio-separator') or 'unknown'}")
        if uvr_mdx.ort is not None:
            print(f"ONNX Runtime: {version('onnxruntime-gpu') or version('onnxruntime') or 'unknown'}")
            print(f"ONNX device: {uvr_mdx.ort.get_device()}")
            print(f"ONNX providers: {uvr_mdx.ort.get_available_providers()}")
            if torch.cuda.is_available() and "CUDAExecutionProvider" in uvr_mdx.ort.get_available_providers():
                print("UVR inference: CUDA")
            else:
                problems.append("UVR CUDA provider unavailable")
        else:
            print(f"ONNX Runtime: unavailable ({type(uvr_mdx.ONNX_RUNTIME_IMPORT_ERROR).__name__}: {uvr_mdx.ONNX_RUNTIME_IMPORT_ERROR})")
            problems.append("ONNX Runtime unavailable")
        print(f"protobuf: {version('protobuf') or 'unknown'}")
        if torch.cuda.is_available():
            print(f"CUDA: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f} GB")
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA build: {torch.version.cuda}")
        else:
            print("CUDA: not detected")
        if not self.ffmpeg:
            problems.append("FFmpeg missing")
        if not seed_vc.INFERENCE_SCRIPT.exists():
            problems.append("Seed-VC missing")
        if uvr_mdx.Separator is None:
            problems.append("audio-separator missing")
        if not uvr_mdx.UVR_INSTRUMENT_MODEL.exists():
            problems.append("Inst_HQ_4 missing")
        if not uvr_mdx.UVR_VOCAL_MODEL.exists():
            problems.append("Voc_FT missing")
        print("Status: " + ("Ready" if not problems else " | ".join(problems)))

    def clear_prev_out(self):
        self.output_path = self.converted_path = None
        self.output_name.configure(text="Show after generate output", text_color="orange")
        self.converted_name.configure(text="Show after generate output", text_color="orange")
        self.out_prev.configure(state="disabled", text="▶")
        self.out_dl.configure(state="disabled")
        self.conv_prev.configure(state="disabled", text="▶")
        self.conv_dl.configure(state="disabled")
        self.inst_dl.configure(state="disabled")
        self.vocal_dl.configure(state="disabled")
        self.bottom_prev.configure(state="disabled", text="▶")
        self.bottom_loop.configure(state="disabled", text="⟲")
        self.bottom_dl.configure(state="disabled")

    def clear_prev_proc(self):
        self.clear_prev_out()
        self.source_wav = self.target_wav = self.seed_source_wav = None
        self.uvr_vocal_path = self.inst_path = self.target_vocal_path = None
        self.tgt_prev_wav = None
        self.inst_name.configure(text="Not separated yet", text_color="gray60")
        self.vocal_name.configure(text="Not separated yet", text_color="gray60")
        self.inst_prev.configure(state="disabled", text="▶")
        self.inst_dl.configure(state="disabled")
        self.vocal_prev.configure(state="disabled", text="▶")
        self.vocal_dl.configure(state="disabled")
        self.upd_config()

    def dl_audio(self, source_path, title, initialfile):
        if not source_path or not Path(source_path).exists():
            return
        path = filedialog.asksaveasfilename(title=title, defaultextension=".wav", filetypes=[("WAV Files", "*.wav")], initialfile=initialfile)
        if path:
            shutil.copy2(source_path, path)
            print(f"Saved: {self._display_path(path)}")
            print("Status: WAV saved")

    def dl_inst(self):
        self.dl_audio(self.inst_path, "Save instrumental WAV", "instrumental_saved.wav")

    def dl_vocal(self):
        self.dl_audio(self.uvr_vocal_path, "Save vocal WAV", "vocal_saved.wav")

    def dl_output(self):
        self.dl_audio(self.output_path, "Save final mix WAV", "final_mix_saved.wav")

    def dl_conv(self):
        self.dl_audio(self.converted_path, "Save converted vocal WAV", "converted_vocal_saved.wav")

    def clear_console(self):
        if self.generating or self.separating or self.target_loading:
            return
        self.console.clear()

    def cleanup_gpu(self):
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def cleanup(self):
        try:
            self.stop_prev()
        except Exception:
            pass
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
        except Exception:
            pass
        self.cleanup_gpu()
        try:
            if self.tts_process and self.tts_process.poll() is None:
                self.tts_process.terminate()
        except Exception:
            pass
        try:
            if self.mic_process and self.mic_process.poll() is None:
                self.mic_process.terminate()
        except Exception:
            pass
        try:
            if getattr(self, "console_stdout", None) is sys.stdout:
                self.console_stdout.flush()
                sys.stdout = self.original_stdout
        except Exception:
            pass
        try:
            shutil.rmtree(self.session_dir, ignore_errors=True)
        except Exception:
            pass

    def close_app(self):
        self.cleanup()
        self.destroy()

    def log_process_output(self, text):
        if text:
            for line in text.splitlines():
                if line.strip():
                    print(line)

    def _display_path(self, path):
        try:
            raw = str(path)
            if not raw:
                return raw

            candidate = os.path.normpath(raw)
            if sys.platform.startswith("win"):
                try:
                    import ctypes
                    long_path = ctypes.create_unicode_buffer(32768)
                    if ctypes.windll.kernel32.GetLongPathNameW(candidate, long_path, len(long_path)):
                        candidate = long_path.value
                except Exception:
                    pass

            candidate = os.path.abspath(candidate)
            temp_root = os.path.normpath(str(TEMP_ROOT))
            user_profile = os.path.normpath(os.environ.get("USERPROFILE", str(Path.home())))
            try:
                rel = os.path.relpath(candidate, temp_root)
                if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
                    return os.path.join("%USERPROFILE%", "AppData", "Local", "Temp", "voicemdx_temp", rel)
            except (ValueError, OSError):
                pass
            try:
                rel = os.path.relpath(candidate, user_profile)
                if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
                    return os.path.join("%USERPROFILE%", rel)
            except (ValueError, OSError):
                pass
            return raw
        except Exception:
            return str(path)

if __name__ == "__main__":
    app = App()
    app.iconbitmap("favicon.ico")
    app.mainloop()