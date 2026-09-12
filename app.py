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
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk
import torch

BASE_DIR = Path(__file__).resolve().parent
UVR_MODEL_DIR = BASE_DIR / "UVR_MODELS"
UVR_INSTRUMENT_MODEL = UVR_MODEL_DIR / "UVR-MDX-NET-Inst_HQ_4.onnx"
UVR_VOCAL_MODEL = UVR_MODEL_DIR / "UVR-MDX-NET-Voc_FT.onnx"
SEED_VC_ROOT = BASE_DIR / "seed-vc"
INFERENCE_SCRIPT = SEED_VC_ROOT / "inference.py"
BIGVGAN_FILE = SEED_VC_ROOT / "modules" / "bigvgan" / "bigvgan.py"
FFMPEG = BASE_DIR / "ffmpeg.exe"
TEMP_ROOT = Path(tempfile.gettempdir()) / "voicemdx_temp"
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".caf"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".ts", ".mts", ".m2ts"]
ALL_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS
MODES = ["Singing", "Speech"]
DIFFUSION_STEP_OPTIONS = {"Low": 25, "Recommended": 50, "High": 75, "Extreme": 100}
FOLLOW_PITCH_OPTIONS = ["Target Voice Pitch", "Source Voice Pitch"]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(str(BASE_DIR / "themes" / "custom.json"))

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

def patch_bigvgan():
    if not BIGVGAN_FILE.exists():
        return "BigVGAN patch skipped: bigvgan.py not found."
    try:
        text = original = BIGVGAN_FILE.read_text(encoding="utf-8")
        text = re.sub(r"(\bproxies:\s*Optional\[Dict\])\s*,", r"\1 = None,", text, count=1)
        text = re.sub(r"(\bresume_download:\s*bool)\s*,", r"\1 = False,", text, count=1)
        if text != original:
            BIGVGAN_FILE.write_text(text, encoding="utf-8")
            return "BigVGAN compatibility patch applied."
        return "BigVGAN compatibility patch ready."
    except Exception as exc:
        return f"BigVGAN patch error: {type(exc).__name__}: {exc}"

try:
    from audio_separator.separator import Separator
    AUDIO_SEPARATOR_IMPORT_ERROR = None
except Exception as exc:
    Separator = None
    AUDIO_SEPARATOR_IMPORT_ERROR = exc

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VoiceMDX - Voice Conversion in Music")
        self.geometry("1240x1080")
        self.minsize(980, 900)
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.mode_var = ctk.StringVar(value="Singing")
        self.steps_var = ctk.IntVar(value=50)
        self.steps_choice_var = ctk.StringVar(value="Recommended")
        self.follow_pitch_var = ctk.StringVar(value="Target Voice Pitch")
        self.source_path = self.target_path = self.output_path = None
        self.source_wav = self.target_wav = self.seed_source_wav = None
        self.uvr_vocal_path = self.instrumental_path = self.target_uvr_vocal_path = None
        self.converted_vocal_path = self.converted_vocal_preview = None
        self.source_preview_wav = self.target_preview_wav = None
        self.process = self.preview_process = self.preview_kind = None
        self.generating = self.separating = self.target_preview_loading = self.separation_complete = False
        self.ffmpeg = str(FFMPEG) if FFMPEG.exists() else None
        self.session_dir = TEMP_ROOT / f"session_{os.getpid()}_{int(time.time())}"
        self.inputs_dir = self.session_dir / "inputs"
        self.preview_dir = self.session_dir / "preview"
        self.instrumental_dir = self.session_dir / "instrumental"
        self.vocal_dir = self.session_dir / "vocal"
        self.target_vocal_dir = self.session_dir / "target_vocal"
        self.normalized_dir = self.session_dir / "normalized"
        self.seed_output_dir = self.session_dir / "seed_output"
        self.output_dir = self.session_dir / "output"
        cleanup_old_sessions(self.session_dir)
        for path in [self.inputs_dir, self.preview_dir, self.instrumental_dir, self.vocal_dir, self.target_vocal_dir, self.normalized_dir, self.seed_output_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.build_ui()
        self.update_config_info()
        self.after(150, self.maximize_window)
        self.log(patch_bigvgan())
        self.check_environment()
        atexit.register(self.cleanup_session)

    def maximize_window(self):
        try:
            self.state("zoomed")
            if sys.platform.startswith("win"):
                self.update_idletasks()
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                ctypes.windll.user32.ShowWindow(hwnd, 3)
                ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception as exc:
            self.log(f"Window maximize warning: {type(exc).__name__}: {exc}")

    def build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        outer = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="nsew", padx=22, pady=18)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        cards = ctk.CTkScrollableFrame(outer, corner_radius=0)
        cards.grid(row=0, column=0, sticky="nsew")
        cards.grid_columnconfigure(0, weight=1)
        cards.grid_columnconfigure(1, weight=1)
        self.cards_scrollable = cards
        self.source_card = self.file_card(cards, 0, "1  Source", "Choose the full song, audio, or video.", self.select_source, "source")
        self.target_card = self.file_card(cards, 0, "2  Target Voice", "Choose the target voice reference.", self.select_target, "target")
        self.source_card["frame"].grid(row=0, column=0, sticky="ew", pady=7, padx=(0, 8))
        self.target_card["frame"].grid(row=0, column=1, sticky="ew", pady=7, padx=(8, 0))
        self.stem_card = self.stem_card_ui(cards, 1)
        self.settings_card = self.settings_ui(cards, 2)
        self.output_card = self.output_ui(cards, 3)
        bottom = ctk.CTkFrame(outer, corner_radius=12)
        bottom.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        bottom.grid_columnconfigure(0, weight=1)
        self.generate_button = ctk.CTkButton(bottom, text="Generate Converted Vocal + Mix", command=self.generate_thread, height=52, font=ctk.CTkFont(size=16, weight="bold"))
        self.generate_button.grid(row=0, column=0, padx=(12, 8), pady=12, sticky="ew")
        self.bottom_preview = ctk.CTkButton(bottom, text="▶", command=self.toggle_output_preview, width=52, height=52, font=ctk.CTkFont(size=18), state="disabled")
        self.bottom_preview.grid(row=0, column=1, padx=4, pady=12)
        self.bottom_download = ctk.CTkButton(bottom, text="⬇️", command=self.download_output, width=46, height=52, font=ctk.CTkFont(size=18), state="disabled")
        self.bottom_download.grid(row=0, column=2, padx=4, pady=12)
        self.clear_button = ctk.CTkButton(bottom, text="Clear", command=self.clear_all, height=52, width=110)
        self.clear_button.grid(row=0, column=3, padx=(8, 12), pady=12)
        self.status_label = ctk.CTkLabel(outer, text="Ready", anchor="w", text_color="gray70")
        self.status_label.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        self.log_box = ctk.CTkTextbox(outer, height=145, wrap="word", font=ctk.CTkFont(family="Consolas", size=11))
        self.log_box.grid(row=3, column=0, sticky="ew")
        self.log_box.configure(state="disabled")

    def file_card(self, parent, row, title, subtitle, command, kind):
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.grid(row=row, column=0, sticky="ew", pady=7)
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=16, pady=(15, 4), sticky="w")
        ctk.CTkLabel(card, text=subtitle, text_color="gray70").grid(row=1, column=0, padx=16, pady=(0, 13), sticky="w")
        name = ctk.CTkLabel(card, text="No file selected", anchor="w", text_color="red")
        name.grid(row=0, column=1, rowspan=2, padx=12, pady=12, sticky="ew")
        ctk.CTkButton(card, text="Choose File", command=command, width=130, height=38).grid(row=0, column=2, padx=(4, 6), pady=12)
        preview = ctk.CTkButton(card, text="▶", command=lambda k=kind: self.toggle_preview(k), width=46, height=38, font=ctk.CTkFont(size=18))
        preview.grid(row=0, column=3, padx=(6, 16), pady=12)
        return {"frame": card, "name": name, "preview": preview}

    def stem_card_ui(self, parent, row):
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.grid(row=row, column=0, columnspan=2, sticky="ew", pady=7)
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="3  UVR Separation", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=(16, 10), pady=(15, 4), sticky="w")
        self.separate_button = ctk.CTkButton(card, text="Separate Vocal & Instrument", command=self.separate_thread, width=205, height=36, font=ctk.CTkFont(size=13, weight="bold"))
        self.separate_button.grid(row=0, column=2, padx=(10, 16), pady=(12, 4), sticky="e")
        self.separation_status = ctk.CTkLabel(card, text="Only Source Voice. If you have instrument on your source, Click Seperate.", text_color="orange")
        self.separation_status.grid(row=1, column=0, columnspan=4, padx=16, pady=(2, 10), sticky="w")
        ctk.CTkLabel(card, text="Instrumental", font=ctk.CTkFont(size=14, weight="bold")).grid(row=2, column=0, padx=16, pady=10, sticky="w")
        self.instrumental_name = ctk.CTkLabel(card, text="Not separated yet", anchor="w", text_color="orange")
        self.instrumental_name.grid(row=2, column=1, padx=12, pady=10, sticky="ew")
        self.instrumental_actions = ctk.CTkFrame(card, fg_color="transparent")
        self.instrumental_actions.grid(row=2, column=2, columnspan=2, padx=(0, 16), pady=10, sticky="e")
        self.instrumental_preview = ctk.CTkButton(self.instrumental_actions, text="▶", command=self.toggle_instrumental_preview, width=46, height=36, font=ctk.CTkFont(size=18), state="disabled")
        self.instrumental_preview.grid(row=0, column=0, padx=(0, 2))
        self.instrumental_download = ctk.CTkButton(self.instrumental_actions, text="⬇️", command=self.download_instrumental, width=46, height=36, font=ctk.CTkFont(size=18), state="disabled")
        self.instrumental_download.grid(row=0, column=1, padx=(2, 0))
        ctk.CTkLabel(card, text="Vocal", font=ctk.CTkFont(size=14, weight="bold")).grid(row=3, column=0, padx=16, pady=(4, 15), sticky="w")
        self.vocal_name = ctk.CTkLabel(card, text="Not separated yet", anchor="w", text_color="orange")
        self.vocal_name.grid(row=3, column=1, padx=12, pady=(4, 15), sticky="ew")
        self.vocal_actions = ctk.CTkFrame(card, fg_color="transparent")
        self.vocal_actions.grid(row=3, column=2, columnspan=2, padx=(0, 16), pady=(4, 15), sticky="e")
        self.vocal_preview = ctk.CTkButton(self.vocal_actions, text="▶", command=self.toggle_vocal_preview, width=46, height=36, font=ctk.CTkFont(size=18), state="disabled")
        self.vocal_preview.grid(row=0, column=0, padx=(0, 2))
        self.vocal_download = ctk.CTkButton(self.vocal_actions, text="⬇️", command=self.download_vocal, width=46, height=36, font=ctk.CTkFont(size=18), state="disabled")
        self.vocal_download.grid(row=0, column=1, padx=(2, 0))
        return card

    def settings_ui(self, parent, row):
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.grid(row=row, column=0, columnspan=2, sticky="ew", pady=7)
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text="Seed-VC Settings", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=16, pady=(14, 7), sticky="w")
        steps_frame = ctk.CTkFrame(card)
        steps_frame.grid(row=1, column=0, padx=16, pady=7, sticky="ew")
        steps_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(steps_frame, text="Voice Quality", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w")
        self.steps_menu = ctk.CTkOptionMenu(steps_frame, variable=self.steps_choice_var, values=list(DIFFUSION_STEP_OPTIONS.keys()), command=self.steps_changed, width=180, height=38)
        self.steps_menu.grid(row=0, column=1, padx=(12, 0), sticky="e")
        mode_frame = ctk.CTkFrame(card)
        mode_frame.grid(row=2, column=0, padx=16, pady=7, sticky="ew")
        mode_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(mode_frame, text="Voice Mode", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w")
        self.mode_menu = ctk.CTkOptionMenu(mode_frame, variable=self.mode_var, values=MODES, command=self.mode_changed, width=180, height=38)
        self.mode_menu.grid(row=0, column=1, padx=(12, 0), sticky="e")
        pitch_frame = ctk.CTkFrame(card)
        pitch_frame.grid(row=3, column=0, padx=16, pady=7, sticky="ew")
        pitch_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(pitch_frame, text="Follow Pitch Voice", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w")
        self.follow_pitch_menu = ctk.CTkOptionMenu(pitch_frame, variable=self.follow_pitch_var, values=FOLLOW_PITCH_OPTIONS, command=self.follow_pitch_changed, width=220, height=38)
        self.follow_pitch_menu.grid(row=0, column=1, padx=(12, 0), sticky="e")
        self.config_info = ctk.CTkLabel(card, text="", text_color="gray70", anchor="w", justify="left")
        self.config_info.grid(row=4, column=0, padx=16, pady=(3, 14), sticky="w")
        return card

    def output_ui(self, parent, row):
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.grid(row=row, column=0, columnspan=2, sticky="ew", pady=7)
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(card, text="4  Final Output", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=16, pady=(15, 4), sticky="w")
        ctk.CTkLabel(card, text="Normalized Instrumental + converted target vocal", text_color="gray70").grid(row=1, column=0, padx=16, pady=(0, 13), sticky="w")
        self.output_name = ctk.CTkLabel(card, text="Show after generate output", anchor="w", text_color="orange")
        self.output_name.grid(row=0, column=1, rowspan=2, padx=12, pady=12, sticky="ew")
        self.output_preview = ctk.CTkButton(card, text="▶", command=self.toggle_output_preview, width=46, height=38, font=ctk.CTkFont(size=18), state="disabled")
        self.output_preview.grid(row=0, column=2, padx=4, pady=12)
        self.output_download = ctk.CTkButton(card, text="⬇️", command=self.download_output, width=46, height=38, font=ctk.CTkFont(size=18), state="disabled")
        self.output_download.grid(row=0, column=3, padx=(4, 16), pady=12)
        self.converted_vocal_name = ctk.CTkLabel(card, text="Show after generate output", anchor="w", text_color="orange")
        self.converted_vocal_name.grid(row=2, column=1, padx=12, pady=(4, 14), sticky="ew")
        ctk.CTkLabel(card, text="Converted Vocal Only", font=ctk.CTkFont(size=14, weight="bold")).grid(row=2, column=0, padx=16, pady=(4, 14), sticky="w")
        self.converted_vocal_preview = ctk.CTkButton(card, text="▶", command=self.toggle_converted_vocal_preview, width=46, height=38, font=ctk.CTkFont(size=18), state="disabled")
        self.converted_vocal_preview.grid(row=2, column=2, padx=4, pady=(4, 14))
        self.converted_vocal_download = ctk.CTkButton(card, text="⬇️", command=self.download_converted_vocal, width=46, height=38, font=ctk.CTkFont(size=18), state="disabled")
        self.converted_vocal_download.grid(row=2, column=3, padx=(4, 16), pady=(4, 14))
        return card

    def mode_changed(self, choice):
        self.mode_var.set(choice)
        self.log(f"Mode selected: {choice}")
        self.update_config_info()

    def steps_changed(self, choice):
        self.steps_var.set(DIFFUSION_STEP_OPTIONS.get(choice, 50))
        self.update_config_info()

    def follow_pitch_changed(self, choice):
        self.follow_pitch_var.set(choice)
        self.update_config_info()

    def update_config_info(self):
        singing = self.mode_var.get() == "Singing"
        target_pitch = self.follow_pitch_var.get() == "Target Voice Pitch"
        descriptions = {(True, True): "Singing in the same voice with minimal pitch change from the source.", (True, False): "Singing in the same voice with the exact same pitch as the source.", (False, True): "Speaking in the exact same voice.", (False, False): "Speaking in the exact voice with minimal pitch change from the source."}
        description = descriptions[(singing, target_pitch)]
        cfg = self.get_config()
        self.config_info.configure(text=f"steps={cfg['steps']} · f0={cfg['f0']} · auto_f0={cfg['auto_f0']}    {description}")

    def choose_file(self, title):
        patterns = " ".join(f"*{ext}" for ext in ALL_EXTENSIONS)
        return filedialog.askopenfilename(title=title, filetypes=[("Audio and Video", patterns), ("All Files", "*.*")])

    def select_source(self):
        if self.generating or self.separating:
            return
        path = self.choose_file("Choose source audio or video")
        if not path:
            return
        self.stop_preview()
        self.source_preview_wav = None
        self.source_path = Path(path)
        self.source_card["name"].configure(text=self.source_path.name, text_color="green")
        self.clear_previous_processing()
        self.separation_complete = False
        self.log(f"Source selected: {self._display_path(self.source_path)}")
        self.set_status("Source selected")

    def select_target(self):
        if self.generating or self.target_preview_loading:
            return
        path = self.choose_file("Choose target voice sample")
        if not path:
            return
        self.stop_preview()
        self.target_preview_wav = None
        self.target_uvr_vocal_path = None
        self.target_path = Path(path)
        self.target_card["name"].configure(text=self.target_path.name, text_color="green")
        self.log(f"Target selected: {self._display_path(self.target_path)}")
        self.set_status("Target selected")

    def separate_thread(self):
        if self.generating or self.separating:
            return
        if not self.source_path:
            self.log("Select a source first.")
            self.set_status("Select source first")
            return
        threading.Thread(target=self.separate_worker, daemon=True).start()

    def separate_worker(self):
        self.separating = True
        self.after(0, lambda: self.separate_button.configure(state="disabled", text="Separating..."))
        try:
            self.ensure_source_stems()
            self.separation_complete = True
            self.after(0, lambda: self.separation_status.configure(text="Seperated Voice and Instrument complete", text_color="green"))
            self.set_status("UVR separation complete")
        except Exception as exc:
            self.log(f"UVR ERROR: {type(exc).__name__}: {exc}")
            self.after(0, lambda e=str(exc): self.separation_status.configure(text=f"Separation failed: {e}", text_color="orange"))
            self.set_status("UVR separation failed")
        finally:
            self.separating = False
            self.after(0, lambda: self.separate_button.configure(state="normal", text="Separate Vocal & Instrument"))

    def ensure_source_stems(self):
        if self.instrumental_path and self.instrumental_path.exists() and self.uvr_vocal_path and self.uvr_vocal_path.exists():
            return
        if not self.source_path or not self.source_path.exists():
            raise RuntimeError("Source audio or video was not found.")
        if not self.ffmpeg:
            raise RuntimeError(f"FFmpeg was not found: {FFMPEG}")
        if Separator is None:
            raise RuntimeError(f"audio-separator import failed: {type(AUDIO_SEPARATOR_IMPORT_ERROR).__name__}: {AUDIO_SEPARATOR_IMPORT_ERROR}")
        if not UVR_INSTRUMENT_MODEL.exists():
            raise RuntimeError(f"Missing instrument model: {UVR_INSTRUMENT_MODEL}")
        if not UVR_VOCAL_MODEL.exists():
            raise RuntimeError(f"Missing vocal model: {UVR_VOCAL_MODEL}")
        self.after(0, lambda: self.separation_status.configure(text="Only Source Voice. If you have instrument on your source, Click Seperate.", text_color="orange"))
        self.source_wav = self.inputs_dir / "source.wav"
        self.extract_audio(self.source_path, self.source_wav, "source", 2)
        self.run_uvr(UVR_INSTRUMENT_MODEL, self.instrumental_dir, "Instrumental")
        self.cleanup_gpu()
        self.run_uvr(UVR_VOCAL_MODEL, self.vocal_dir, "Vocals")
        self.cleanup_gpu()
        self.instrumental_path = self.normalize_audio(self.instrumental_path, "instrumental")
        self.uvr_vocal_path = self.normalize_audio(self.uvr_vocal_path, "source_vocal")
        self.separation_complete = True
        self.after(0, lambda: self.instrumental_name.configure(text=self.instrumental_path.name, text_color="green"))
        self.after(0, lambda: self.vocal_name.configure(text=self.uvr_vocal_path.name, text_color="green"))
        self.after(0, lambda: self.separation_status.configure(text="Seperated Voice and Instrument complete", text_color="green"))
        self.after(0, lambda: self.instrumental_preview.configure(state="normal"))
        self.after(0, lambda: self.instrumental_download.configure(state="normal"))
        self.after(0, lambda: self.vocal_preview.configure(state="normal"))
        self.after(0, lambda: self.vocal_download.configure(state="normal"))
        self.log(f"Normalized instrumental: {self._display_path(self.instrumental_path)}")
        self.log(f"Normalized source vocal: {self._display_path(self.uvr_vocal_path)}")

    def run_uvr(self, model_path, output_dir, stem_name):
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in output_dir.glob("*.wav"):
            try:
                item.unlink()
            except Exception:
                pass
        self.log(f"Loading {model_path.name}")
        separator = Separator(output_dir=str(output_dir), model_file_dir=str(UVR_MODEL_DIR), output_format="WAV", sample_rate=44100, use_soundfile=True, use_autocast=False, output_single_stem=stem_name)
        separator.load_model(model_filename=model_path.name)
        result = separator.separate(str(self.source_wav))
        paths = self.flatten_paths(result)
        discovered = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        paths += [p for p in discovered if p not in paths]
        selected = self.pick_stem_path(paths, stem_name) or (paths[0] if paths else None)
        if not selected or not selected.exists():
            raise RuntimeError(f"{model_path.name} did not produce {stem_name}.wav")
        if stem_name.lower() == "instrumental":
            self.instrumental_path = selected
        else:
            self.uvr_vocal_path = selected
        self.log(f"{stem_name}: {self._display_path(selected)}")

    def run_target_uvr(self):
        if not self.target_path or not self.target_path.exists():
            raise RuntimeError("Target voice file not found.")
        if not self.ffmpeg:
            raise RuntimeError(f"FFmpeg was not found: {FFMPEG}")
        if Separator is None:
            raise RuntimeError(f"audio-separator import failed: {type(AUDIO_SEPARATOR_IMPORT_ERROR).__name__}: {AUDIO_SEPARATOR_IMPORT_ERROR}")
        if not UVR_VOCAL_MODEL.exists():
            raise RuntimeError(f"Missing vocal model: {UVR_VOCAL_MODEL}")
        target_input = self.inputs_dir / "target.wav"
        self.extract_audio(self.target_path, target_input, "target", 1)
        self.target_vocal_dir.mkdir(parents=True, exist_ok=True)
        for item in self.target_vocal_dir.glob("*.wav"):
            try:
                item.unlink()
            except Exception:
                pass
        self.log(f"Cleaning target with {UVR_VOCAL_MODEL.name}")
        separator = Separator(output_dir=str(self.target_vocal_dir), model_file_dir=str(UVR_MODEL_DIR), output_format="WAV", sample_rate=44100, use_soundfile=True, use_autocast=False, output_single_stem="Vocals")
        separator.load_model(model_filename=UVR_VOCAL_MODEL.name)
        result = separator.separate(str(target_input))
        paths = self.flatten_paths(result)
        discovered = sorted(self.target_vocal_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        paths += [p for p in discovered if p not in paths]
        selected = self.pick_stem_path(paths, "Vocals") or (paths[0] if paths else None)
        if not selected or not selected.exists():
            raise RuntimeError("Voc_FT did not produce a cleaned target vocal.")
        self.target_uvr_vocal_path = self.normalize_audio(selected, "target_vocal")
        self.target_wav = self.target_preview_wav = self.target_uvr_vocal_path
        self.log(f"Clean target vocal: {self._display_path(self.target_uvr_vocal_path)}")
        self.cleanup_gpu()
        return self.target_uvr_vocal_path

    def flatten_paths(self, value):
        if isinstance(value, (str, Path)):
            return [Path(value)]
        if isinstance(value, dict):
            value = value.values()
        if isinstance(value, (list, tuple, set)):
            result = []
            for item in value:
                result.extend(self.flatten_paths(item))
            return result
        return []

    def pick_stem_path(self, paths, name):
        key = name.lower()
        return next((p for p in paths if p.exists() and key in p.stem.lower()), None)

    def normalize_audio(self, input_path, name):
        if not self.ffmpeg:
            raise RuntimeError(f"FFmpeg was not found: {FFMPEG}")
        output = self.normalized_dir / f"{Path(input_path).stem}_{name}.wav"
        command = [self.ffmpeg, "-y", "-i", str(input_path), "-af", "loudnorm=I=-16:LRA=11:TP=-1", "-ar", "44100", "-c:a", "pcm_s16le", str(output)]
        self.log(f"Normalizing: {Path(input_path).name}")
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(f"Loudness normalization failed for {input_path}.")
        return output

    def extract_audio(self, input_path, output_path, label, channels=1):
        if not self.ffmpeg:
            raise RuntimeError(f"FFmpeg was not found: {FFMPEG}")
        command = [self.ffmpeg, "-y", "-i", str(input_path), "-vn", "-ac", str(channels), "-ar", "44100", "-c:a", "pcm_s16le", str(output_path)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not output_path.exists():
            self.log_process_output(completed.stdout)
            raise RuntimeError(f"FFmpeg failed to extract {label} audio.")
        self.log(f"{label.title()} audio ready: {output_path}")

    def prepare_seed_source(self, source_path=None):
        self.seed_source_wav = self.inputs_dir / "seed_source.wav"
        input_source = source_path or self.uvr_vocal_path
        self.extract_audio(input_source, self.seed_source_wav, "source vocal", 1)
        self.seed_source_wav = self.normalize_audio(self.seed_source_wav, "seed_source")

    def get_config(self):
        steps = DIFFUSION_STEP_OPTIONS.get(self.steps_choice_var.get(), 50)
        target_pitch = self.follow_pitch_var.get() == "Target Voice Pitch"
        singing = self.mode_var.get() == "Singing"
        return {"steps": steps, "cfg": 0.80, "f0": singing, "auto_f0": target_pitch if singing else not target_pitch, "pitch": 0}

    def seed_command(self):
        cfg = self.get_config()
        return [sys.executable, str(INFERENCE_SCRIPT), "--source", str(self.seed_source_wav), "--target", str(self.target_wav), "--output", str(self.seed_output_dir), "--diffusion-steps", str(cfg["steps"]), "--length-adjust", "1.0", "--inference-cfg-rate", str(cfg["cfg"]), "--f0-condition", str(cfg["f0"]), "--auto-f0-adjust", str(cfg["auto_f0"]), "--semi-tone-shift", "0", "--fp16", "True"]

    def generate_thread(self):
        if self.generating or self.separating or self.target_preview_loading:
            return
        threading.Thread(target=self.generate, daemon=True).start()

    def generate(self):
        if not self.source_path or not self.target_path:
            self.log("Source and target are required.")
            self.set_status("Select source and target")
            return
        if not self.ffmpeg or not INFERENCE_SCRIPT.exists():
            self.check_environment()
            return
        self.generating = True
        self.after(0, lambda: self.generate_button.configure(state="disabled", text="Converting + Mixing..."))
        try:
            self.stop_preview()
            if self.separation_complete:
                self.set_status("Preparing separated source vocal and instrumental...")
                self.ensure_source_stems()
                seed_input = self.uvr_vocal_path
            else:
                self.set_status("Using source as vocal only - skipping UVR separation and final mixing...")
                seed_input = self.source_path
            self.set_status("Cleaning target voice...")
            self.target_wav = self.target_uvr_vocal_path = self.run_target_uvr()
            self.prepare_seed_source(seed_input)
            cfg = self.get_config()
            self.log(f"Seed-VC steps: {cfg['steps']}")
            self.log(f"Seed-VC strength: {cfg['cfg']:.2f}")
            self.log(f"Seed-VC F0: {cfg['f0']}")
            self.log(f"Seed-VC Follow Pitch Voice: {self.follow_pitch_var.get()}")
            self.set_status("Running Seed-VC...")
            self.run_seed_vc()
            self.converted_vocal_path = self.normalize_audio(self.converted_vocal_path, "converted_vocal")
            self.soften_converted_vocal()
            if self.separation_complete:
                self.set_status("Mixing final output...")
                self.mix_final()
            else:
                self.output_path = self.converted_vocal_path
            self.after(0, lambda: self.output_name.configure(text=self.output_path.name, text_color="green"))
            self.after(0, lambda: self.converted_vocal_name.configure(text=self.converted_vocal_path.name, text_color="green"))
            self.after(0, lambda: self.output_preview.configure(state="normal", text="▶"))
            self.after(0, lambda: self.output_download.configure(state="normal"))
            self.after(0, lambda: self.converted_vocal_preview.configure(state="normal", text="▶"))
            self.after(0, lambda: self.converted_vocal_download.configure(state="normal"))
            self.after(0, lambda: self.bottom_preview.configure(state="normal", text="▶"))
            self.after(0, lambda: self.bottom_download.configure(state="normal"))
            self.set_status("Conversion complete")
            self.log(f"Final output: {self._display_path(self.output_path)}")
        except Exception as exc:
            self.log(f"GENERATION ERROR: {type(exc).__name__}: {exc}")
            self.set_status("Generation failed")
        finally:
            self.cleanup_gpu()
            self.generating = False
            self.after(0, lambda: self.generate_button.configure(state="normal", text="Generate Converted Vocal + Mix"))

    def run_seed_vc(self):
        for item in self.seed_output_dir.glob("*.wav"):
            try:
                item.unlink()
            except Exception:
                pass
        self.process = subprocess.Popen(self.seed_command(), cwd=str(SEED_VC_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=os.environ.copy())
        for line in iter(self.process.stdout.readline, ""):
            if line:
                self.log(line.rstrip())
        self.process.stdout.close()
        code = self.process.wait()
        self.process = None
        if code != 0:
            raise RuntimeError(f"Seed-VC exited with code {code}.")
        outputs = sorted(self.seed_output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not outputs:
            raise RuntimeError("Seed-VC produced no WAV output.")
        self.converted_vocal_path = outputs[0]
        self.log(f"Converted vocal: {self._display_path(self.converted_vocal_path)}")

    def soften_converted_vocal(self):
        if not self.ffmpeg or not self.converted_vocal_path or not Path(self.converted_vocal_path).exists():
            raise RuntimeError("Converted vocal was not found for softening.")
        output = self.normalized_dir / "converted_vocal_softened.wav"
        filter_chain = "highpass=f=70,lowpass=f=14500,equalizer=f=6200:t=q:w=1.0:g=-3.0,equalizer=f=9000:t=q:w=1.2:g=-2.0,acompressor=threshold=-18dB:ratio=2:attack=8:release=100,alimiter=limit=-2dB"
        command = [self.ffmpeg, "-y", "-i", str(self.converted_vocal_path), "-af", filter_chain, "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(output)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not output.exists():
            self.log_process_output(completed.stdout)
            raise RuntimeError("Converted vocal softening failed.")
        self.converted_vocal_path = output
        self.log(f"Softened converted vocal: {self._display_path(self.converted_vocal_path)}")

    def mix_final(self):
        self.output_path = self.output_dir / "final_mix.wav"
        filter_complex = chr(59).join(["[0:a]aresample=44100[a0]", f"[1:a]aresample=44100,volume=5dB[a1]", "[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0:normalize=1[mix]", "[mix]loudnorm=I=-16:LRA=11:TP=-1.5[out]"])
        command = [self.ffmpeg, "-y", "-i", str(self.instrumental_path), "-i", str(self.converted_vocal_path), "-filter_complex", filter_complex, "-map", "[out]", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(self.output_path)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not self.output_path.exists():
            self.log_process_output(completed.stdout)
            raise RuntimeError("Final mix failed.")
        self.log(f"Final mix created: {self._display_path(self.output_path)}")

    def toggle_source_preview(self):
        self.toggle_preview("source")

    def toggle_target_preview(self):
        self.toggle_preview("target")

    def toggle_instrumental_preview(self):
        self.toggle_preview("instrumental")

    def toggle_vocal_preview(self):
        self.toggle_preview("vocal")

    def toggle_output_preview(self):
        self.toggle_preview("output")

    def toggle_converted_vocal_preview(self):
        self.toggle_preview("converted_vocal")

    def toggle_preview(self, kind):
        if self.preview_kind == kind:
            self.stop_preview()
            return
        if kind == "source":
            self.play_preview_path(self.source_preview_wav or self.prepare_preview(self.source_path, "source"), kind)
            return
        if kind == "target":
            if self.target_uvr_vocal_path and self.target_uvr_vocal_path.exists():
                self.target_preview_wav = self.target_uvr_vocal_path
                self.play_preview_path(self.target_preview_wav, kind)
                return
            if not self.target_path or not self.target_path.exists():
                self.log("Preview unavailable: target")
                return
            if self.target_preview_loading:
                return
            self.target_preview_loading = True
            self.after(0, lambda: self.target_card["preview"].configure(state="disabled", text="…"))
            self.set_status("Cleaning target for preview...")
            threading.Thread(target=self.prepare_target_preview_worker, daemon=True).start()
            return
        paths = {"instrumental": self.instrumental_path, "vocal": self.uvr_vocal_path, "converted_vocal": self.converted_vocal_path}
        self.play_preview_path(paths.get(kind, self.output_path), kind)

    def prepare_target_preview_worker(self):
        try:
            target = self.run_target_uvr()
            self.after(0, lambda: self.target_card["preview"].configure(state="normal", text="▶"))
            self.set_status("Target vocal ready")
            self.after(0, lambda p=target: self.play_preview_path(p, "target"))
        except Exception as exc:
            self.log(f"TARGET PREVIEW ERROR: {type(exc).__name__}: {exc}")
            self.set_status("Target preview failed")
            self.after(0, lambda: self.target_card["preview"].configure(state="normal", text="▶"))
        finally:
            self.target_preview_loading = False

    def play_preview_path(self, path, kind):
        if not path or not Path(path).exists():
            self.log(f"Preview unavailable: {kind}")
            return
        self.stop_preview()
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                self.preview_process = "winsound"
            else:
                ffplay = shutil.which("ffplay")
                if not ffplay:
                    raise RuntimeError("ffplay was not found.")
                self.preview_process = subprocess.Popen([ffplay, "-nodisp", "-autoexit", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.preview_kind = kind
            self.set_preview_icons(kind)
            self.log(f"Playing {kind} preview: {self._display_path(path)}")
        except Exception as exc:
            self.preview_process = self.preview_kind = None
            self.set_preview_icons()
            self.log(f"Preview error: {type(exc).__name__}: {exc}")

    def prepare_preview(self, source_path, kind):
        if not source_path or not Path(source_path).exists() or not self.ffmpeg:
            return None
        output = self.preview_dir / f"{kind}.wav"
        command = [self.ffmpeg, "-y", "-i", str(source_path), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(output)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not output.exists():
            return None
        if kind == "source":
            self.source_preview_wav = output
        return output

    def set_preview_icons(self, active=None):
        buttons = [("source", self.source_card["preview"]), ("target", self.target_card["preview"]), ("instrumental", self.instrumental_preview), ("vocal", self.vocal_preview), ("output", self.output_preview), ("converted_vocal", self.converted_vocal_preview), ("output", self.bottom_preview)]
        for kind, button in buttons:
            button.configure(text="■" if kind == active else "▶")

    def stop_preview(self):
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
        if hasattr(self, "source_card"):
            self.set_preview_icons()

    def check_environment(self):
        problems = []
        self.log(f"FFmpeg: {self.ffmpeg or f'missing: {FFMPEG}'}")
        self.log(f"Seed-VC: {'found' if INFERENCE_SCRIPT.exists() else 'missing'}")
        self.log(f"Inst_HQ_4: {'found' if UVR_INSTRUMENT_MODEL.exists() else 'missing'}")
        self.log(f"Voc_FT: {'found' if UVR_VOCAL_MODEL.exists() else 'missing'}")
        self.log(f"audio-separator: {'available' if Separator is not None else 'missing'}")
        self.log(f"protobuf: {version('protobuf') or 'unknown'}")
        if torch.cuda.is_available():
            self.log(f"CUDA: {torch.cuda.get_device_name(0)}")
            self.log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f} GB")
            self.log(f"PyTorch: {torch.__version__}")
            self.log(f"CUDA build: {torch.version.cuda}")
        else:
            self.log("CUDA: not detected")
        if not self.ffmpeg:
            problems.append("FFmpeg missing")
        if not INFERENCE_SCRIPT.exists():
            problems.append("Seed-VC missing")
        if Separator is None:
            problems.append("audio-separator missing")
        if not UVR_INSTRUMENT_MODEL.exists():
            problems.append("Inst_HQ_4 missing")
        if not UVR_VOCAL_MODEL.exists():
            problems.append("Voc_FT missing")
        self.set_status("Ready" if not problems else " | ".join(problems))

    def clear_previous_output_only(self):
        self.output_path = self.converted_vocal_path = None
        self.output_name.configure(text="Show after generate output", text_color="orange")
        self.converted_vocal_name.configure(text="Show after generate output", text_color="orange")
        self.output_preview.configure(state="disabled", text="▶")
        self.output_download.configure(state="disabled")
        self.converted_vocal_preview.configure(state="disabled", text="▶")
        self.converted_vocal_download.configure(state="disabled")
        self.instrumental_download.configure(state="disabled")
        self.vocal_download.configure(state="disabled")
        self.bottom_preview.configure(state="disabled", text="▶")
        self.bottom_download.configure(state="disabled")

    def clear_previous_processing(self):
        self.clear_previous_output_only()
        self.source_wav = self.target_wav = self.seed_source_wav = None
        self.uvr_vocal_path = self.instrumental_path = self.target_uvr_vocal_path = None
        self.target_preview_wav = None
        self.separation_complete = False
        self.instrumental_name.configure(text="Not separated yet", text_color="orange")
        self.vocal_name.configure(text="Not separated yet", text_color="orange")
        self.instrumental_preview.configure(state="disabled", text="▶")
        self.instrumental_download.configure(state="disabled")
        self.vocal_preview.configure(state="disabled", text="▶")
        self.vocal_download.configure(state="disabled")
        self.separation_status.configure(text="Only Source Voice. If you have instrument on your source, Click Seperate.", text_color="orange")

    def download_audio(self, source_path, title, initialfile):
        if not source_path or not Path(source_path).exists():
            return
        path = filedialog.asksaveasfilename(title=title, defaultextension=".wav", filetypes=[("WAV Files", "*.wav")], initialfile=initialfile)
        if path:
            shutil.copy2(source_path, path)
            self.log(f"Saved: {self._display_path(path)}")
            self.set_status("WAV saved")

    def download_instrumental(self):
        self.download_audio(self.instrumental_path, "Save instrumental WAV", "instrumental_saved.wav")

    def download_vocal(self):
        self.download_audio(self.uvr_vocal_path, "Save vocal WAV", "vocal_saved.wav")

    def download_output(self):
        self.download_audio(self.output_path, "Save final mix WAV", "final_mix_saved.wav")

    def download_converted_vocal(self):
        self.download_audio(self.converted_vocal_path, "Save converted vocal WAV", "converted_vocal_saved.wav")

    def clear_all(self):
        if self.generating or self.separating or self.target_preview_loading:
            return
        self.stop_preview()
        self.source_path = self.target_path = None
        self.source_card["name"].configure(text="No file selected", text_color="red")
        self.target_card["name"].configure(text="No file selected", text_color="red")
        self.clear_previous_processing()
        self.steps_var.set(50)
        self.steps_choice_var.set("Recommended")
        self.mode_var.set("Singing")
        self.follow_pitch_var.set("Target Voice Pitch")
        self.update_config_info()
        self.set_status("Ready")
        self.log("Cleared.")

    def cleanup_gpu(self):
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def cleanup_session(self):
        try:
            self.stop_preview()
        except Exception:
            pass
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
        except Exception:
            pass
        self.cleanup_gpu()
        try:
            shutil.rmtree(self.session_dir, ignore_errors=True)
        except Exception:
            pass

    def close_app(self):
        self.cleanup_session()
        self.destroy()

    def log_process_output(self, text):
        if text:
            for line in text.splitlines():
                if line.strip():
                    self.log(line)

    def _display_path(self, path):
        try:
            raw = str(path)
            normalized = os.path.normpath(raw)
            temp_root = os.path.normpath(str(TEMP_ROOT))
            user_profile = os.path.normpath(os.environ.get("USERPROFILE", str(Path.home())))
            try:
                rel = os.path.relpath(normalized, temp_root)
                if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
                    return os.path.join("%USERPROFILE%", "AppData", "Local", "Temp", "voicemdx_temp", rel)
            except (ValueError, OSError):
                pass
            try:
                rel = os.path.relpath(normalized, user_profile)
                if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
                    return os.path.join("%USERPROFILE%", rel)
            except (ValueError, OSError):
                pass
            return raw
        except Exception:
            return str(path)

    def _display_log_text(self, text):
        value = str(text)
        temp_prefix = os.path.normpath(str(TEMP_ROOT)).rstrip("\\/")
        if temp_prefix:
            value = re.sub(re.escape(temp_prefix) + r"(?=[\\/]|$)", lambda m: os.path.join("%USERPROFILE%", "AppData", "Local", "Temp", "voicemdx_temp"), value, flags=re.IGNORECASE)
        profile_prefix = os.path.normpath(os.environ.get("USERPROFILE", str(Path.home()))).rstrip("\\/")
        if profile_prefix:
            value = re.sub(re.escape(profile_prefix) + r"(?=[\\/]|$)", "%USERPROFILE%", value, flags=re.IGNORECASE)
        return value

    def log(self, text):
        display_text = self._display_log_text(text)
        print(display_text, flush=True)
        try:
            self.after(0, lambda t=display_text: self._append_log(t))
        except Exception:
            pass

    def _append_log(self, text):
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", text + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        except Exception:
            pass

    def set_status(self, text):
        try:
            self.after(0, lambda t=str(text): self.status_label.configure(text=t))
        except Exception:
            pass

if __name__ == "__main__":
    app = App()
    app.iconbitmap("favicon.ico")
    app.mainloop()