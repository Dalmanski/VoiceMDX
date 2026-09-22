import os
import sys
import gc
import json
import shutil
import threading
import tempfile
import traceback
import subprocess
import winsound
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog
import numpy as np

SUGGESTED_TARGET_LYRICS = "<SP> We just met here today <SP> And now this feels crazy <SP> But here is my number <SP> Call <SP> Maybe you can call <SP> Hard to find the words <SP> That is so funny <SP> So please call me"

if int(np.__version__.split(".")[0]) >= 2:
    raise RuntimeError(f"NumPy {np.__version__} is not compatible with this SoulX-Singer PyTorch environment. Install NumPy 1.26.4.")

import librosa
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SOULXSINGER_ROOT = r"Z:\Models\SoulX_Singer"
FFMPEG_PATH = PROJECT_ROOT / "ffmpeg.exe"
PARAKEET_MODEL_PATH = Path(SOULXSINGER_ROOT) / "pretrained_models" / "SoulX-Singer-Preprocess" / "parakeet-tdt-0.6b-v2" / "parakeet-tdt-0.6b-v2.nemo"
os.environ["PATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if not FFMPEG_PATH.is_file():
    raise FileNotFoundError(f"FFmpeg executable was not found at {FFMPEG_PATH}")

if not os.path.isdir(SOULXSINGER_ROOT):
    raise RuntimeError(f"SoulX-Singer repository was not found at: {SOULXSINGER_ROOT}")

required_paths = {"preprocess.pipeline": Path(SOULXSINGER_ROOT) / "preprocess" / "pipeline.py", "preprocess.tools.g2p": Path(SOULXSINGER_ROOT) / "preprocess" / "tools" / "g2p.py", "preprocess.tools.lyric_transcription": Path(SOULXSINGER_ROOT) / "preprocess" / "tools" / "lyric_transcription.py", "cli.inference": Path(SOULXSINGER_ROOT) / "cli" / "inference.py", "soulxsinger.config": Path(SOULXSINGER_ROOT) / "soulxsinger" / "config" / "soulxsinger.yaml", "soulxsinger.phoneset": Path(SOULXSINGER_ROOT) / "soulxsinger" / "utils" / "phoneme" / "phone_set.json", "SoulX-Singer model": Path(SOULXSINGER_ROOT) / "pretrained_models" / "SoulX-Singer" / "model.pt", "SoulX-Singer Parakeet model": PARAKEET_MODEL_PATH}

missing_paths = [f"{name}: {path}" for name, path in required_paths.items() if not path.is_file()]

if missing_paths:
    missing_text = "\n".join(missing_paths)
    raise RuntimeError(f"SoulX-Singer installation is incomplete. Missing required files:\n{missing_text}\nClone the official repository into {SOULXSINGER_ROOT} and keep the pretrained_models folders there.")

sys.path.insert(0, SOULXSINGER_ROOT)
os.chdir(SOULXSINGER_ROOT)

try:
    from preprocess.pipeline import PreprocessPipeline
    from preprocess.tools.g2p import g2p_transform
    from soulxsinger.utils.file_utils import load_config
    from cli.inference import build_model as build_svs_model
    from cli.inference import process as svs_process
except Exception as exc:
    traceback.print_exc()
    raise RuntimeError(f"Failed to import SoulX-Singer components from {SOULXSINGER_ROOT}: {exc}") from exc

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SoulXSingerLyricsGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SoulX-Singer Lyric Editor")
        self.geometry("1150x930")
        self.minsize(850, 700)
        self.song_path = ctk.StringVar()
        self.output_path = ctk.StringVar(value=str(Path(SOULXSINGER_ROOT) / "outputs" / "edited_song.wav"))
        self.language = ctk.StringVar(value="English")
        self.control = ctk.StringVar(value="score")
        self.pitch_shift = ctk.IntVar(value=0)
        self.auto_shift = ctk.BooleanVar(value=True)
        self.fp16 = ctk.BooleanVar(value=True)
        self.status = ctk.StringVar(value="Ready")
        self.progress_value = ctk.DoubleVar(value=0.0)
        self.original_token_counter = ctk.StringVar(value="Singing: 0 | Pauses: 0 | Total: 0 | Pause positions: -")
        self.target_token_counter = ctk.StringVar(value="Singing: 0 | Pauses: 0 | Total: 0 | Pause positions: - | Required: 0/0/0 | MATCH")
        self.original_token_layout = None
        self.vocal_path = None
        self.instrumental_path = None
        self.converted_audio_path = None
        self.original_metadata_path = None
        self.target_metadata_path = None
        self.prompt_metadata_path = None
        self.session_dir = None
        self.generated_final_path = None
        self.svs_model = None
        self.svs_config = None
        self.preprocess_pipeline = None
        self.previewing_source = False
        self.previewing_output = False
        self.is_busy = False
        self.create_ui()
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.verify_cuda()

    def create_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        root = ctk.CTkFrame(self, corner_radius=0)
        root.grid(row=0, column=0, sticky="nsew")
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=0)
        root.grid_rowconfigure(2, weight=1)
        header = ctk.CTkFrame(root)
        header.grid(row=0, column=0, padx=14, pady=(14, 8), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(header, text="SoulX-Singer Lyric Editor", font=ctk.CTkFont(size=28, weight="bold"))
        title.grid(row=0, column=0, padx=18, pady=(14, 2), sticky="w")
        subtitle = ctk.CTkLabel(header, text="Edit lyrics while preserving the exact SoulX <SP>/<AP> token layout.", font=ctk.CTkFont(size=14))
        subtitle.grid(row=1, column=0, padx=18, pady=(0, 14), sticky="w")
        status_bar = ctk.CTkFrame(root)
        status_bar.grid(row=1, column=0, padx=14, pady=8, sticky="ew")
        status_bar.grid_columnconfigure(0, weight=1)
        status_label = ctk.CTkLabel(status_bar, textvariable=self.status, anchor="w", font=ctk.CTkFont(size=13, weight="bold"))
        status_label.grid(row=0, column=0, padx=14, pady=10, sticky="ew")
        columns = ctk.CTkFrame(root, fg_color="transparent")
        columns.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="nsew")
        columns.grid_columnconfigure(0, weight=1, uniform="columns")
        columns.grid_columnconfigure(1, weight=1, uniform="columns")
        columns.grid_rowconfigure(0, weight=1)
        left_scroll = ctk.CTkScrollableFrame(columns, label_text="Source & Settings")
        left_scroll.grid(row=0, column=0, padx=(0, 7), sticky="nsew")
        left_scroll.grid_columnconfigure(0, weight=1)
        right_scroll = ctk.CTkScrollableFrame(columns, label_text="Lyrics & Generation")
        right_scroll.grid(row=0, column=1, padx=(7, 0), sticky="nsew")
        right_scroll.grid_columnconfigure(0, weight=1)
        input_frame = ctk.CTkFrame(left_scroll)
        input_frame.grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        input_title = ctk.CTkLabel(input_frame, text="Audio", font=ctk.CTkFont(size=18, weight="bold"))
        input_title.grid(row=0, column=0, columnspan=4, padx=14, pady=(14, 8), sticky="w")
        song_label = ctk.CTkLabel(input_frame, text="Input Audio / Video", anchor="w")
        song_label.grid(row=1, column=0, padx=12, pady=10, sticky="w")
        song_entry = ctk.CTkEntry(input_frame, textvariable=self.song_path)
        song_entry.grid(row=1, column=1, padx=8, pady=10, sticky="ew")
        song_button = ctk.CTkButton(input_frame, text="Browse", width=90, command=self.browse_song)
        song_button.grid(row=1, column=2, padx=5, pady=10)
        self.source_play_button = ctk.CTkButton(input_frame, text="▶ Play", width=90, command=self.toggle_source_playback)
        self.source_play_button.grid(row=1, column=3, padx=(0, 10), pady=10)
        output_label = ctk.CTkLabel(input_frame, text="Output WAV", anchor="w")
        output_label.grid(row=2, column=0, padx=12, pady=10, sticky="w")
        output_entry = ctk.CTkEntry(input_frame, textvariable=self.output_path)
        output_entry.grid(row=2, column=1, padx=8, pady=10, sticky="ew")
        output_button = ctk.CTkButton(input_frame, text="Save As", width=90, command=self.browse_output)
        output_button.grid(row=2, column=2, padx=5, pady=10)
        self.output_play_button = ctk.CTkButton(input_frame, text="▶ Play", width=90, command=self.toggle_output_playback)
        self.output_play_button.grid(row=2, column=3, padx=(0, 10), pady=10)
        settings_frame = ctk.CTkFrame(left_scroll)
        settings_frame.grid(row=1, column=0, padx=8, pady=8, sticky="ew")
        settings_frame.grid_columnconfigure(1, weight=1)
        settings_title = ctk.CTkLabel(settings_frame, text="SoulX Settings", font=ctk.CTkFont(size=18, weight="bold"))
        settings_title.grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w")
        language_label = ctk.CTkLabel(settings_frame, text="Lyric Language", anchor="w")
        language_label.grid(row=1, column=0, padx=12, pady=10, sticky="w")
        language_menu = ctk.CTkOptionMenu(settings_frame, values=["English", "Mandarin", "Cantonese"], variable=self.language)
        language_menu.grid(row=1, column=1, padx=12, pady=10, sticky="ew")
        control_label = ctk.CTkLabel(settings_frame, text="Control", anchor="w")
        control_label.grid(row=2, column=0, padx=12, pady=10, sticky="w")
        control_menu = ctk.CTkOptionMenu(settings_frame, values=["melody", "score"], variable=self.control)
        control_menu.grid(row=2, column=1, padx=12, pady=10, sticky="ew")
        pitch_label = ctk.CTkLabel(settings_frame, text="Pitch Shift", anchor="w")
        pitch_label.grid(row=3, column=0, padx=12, pady=10, sticky="w")
        pitch_slider = ctk.CTkSlider(settings_frame, from_=-24, to=24, number_of_steps=48, variable=self.pitch_shift)
        pitch_slider.grid(row=3, column=1, padx=12, pady=10, sticky="ew")
        pitch_value = ctk.CTkLabel(settings_frame, textvariable=self.pitch_shift, anchor="w")
        pitch_value.grid(row=4, column=0, padx=12, pady=10, sticky="w")
        self.auto_shift_check = ctk.CTkCheckBox(settings_frame, text="Auto Pitch Shift", variable=self.auto_shift)
        self.auto_shift_check.grid(row=4, column=1, padx=12, pady=10, sticky="w")
        self.fp16_check = ctk.CTkCheckBox(settings_frame, text="FP16", variable=self.fp16)
        self.fp16_check.grid(row=5, column=0, padx=12, pady=10, sticky="w")
        instruction_label = ctk.CTkLabel(settings_frame, text="Each normal word is one singing token. <SP> and <AP> are pause markers and must stay in exactly the same token positions.", anchor="w", wraplength=430, justify="left")
        instruction_label.grid(row=5, column=1, padx=12, pady=10, sticky="w")
        original_frame = ctk.CTkFrame(left_scroll)
        original_frame.grid(row=2, column=0, padx=8, pady=8, sticky="ew")
        original_frame.grid_columnconfigure(0, weight=1)
        original_title = ctk.CTkLabel(original_frame, text="Original Lyrics", font=ctk.CTkFont(size=18, weight="bold"))
        original_title.grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")
        original_help = ctk.CTkLabel(original_frame, text="This is the exact layout SoulX detected. Use it as the reference for pause placement.", anchor="w", wraplength=430, justify="left")
        original_help.grid(row=1, column=0, padx=14, pady=(0, 6), sticky="w")
        self.original_text = ctk.CTkTextbox(original_frame, height=180, wrap="word")
        self.original_text.grid(row=2, column=0, padx=14, pady=6, sticky="ew")
        self.original_text.configure(state="disabled")
        self.original_token_label = ctk.CTkLabel(original_frame, textvariable=self.original_token_counter, anchor="w", justify="left", wraplength=430)
        self.original_token_label.grid(row=3, column=0, padx=14, pady=(4, 14), sticky="w")
        target_frame = ctk.CTkFrame(right_scroll)
        target_frame.grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        target_frame.grid_columnconfigure(0, weight=1)
        target_title = ctk.CTkLabel(target_frame, text="New Lyrics", font=ctk.CTkFont(size=18, weight="bold"))
        target_title.grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")
        target_help = ctk.CTkLabel(target_frame, text="Replace the words, but keep the <SP>/<AP> markers in the exact positions shown by the Original Lyrics.", anchor="w", wraplength=430, justify="left")
        target_help.grid(row=1, column=0, padx=14, pady=(0, 6), sticky="w")
        self.target_text = ctk.CTkTextbox(target_frame, height=220, wrap="word")
        self.target_text.grid(row=2, column=0, padx=14, pady=6, sticky="ew")
        self.target_text.bind("<KeyRelease>", self.update_token_counters)
        self.target_token_label = ctk.CTkLabel(target_frame, textvariable=self.target_token_counter, anchor="w", justify="left", wraplength=430)
        self.target_token_label.grid(row=3, column=0, padx=14, pady=(4, 10), sticky="w")
        validate_button = ctk.CTkButton(target_frame, text="Validate Lyrics", height=42, command=self.validate_current_lyrics)
        validate_button.grid(row=4, column=0, padx=14, pady=6, sticky="ew")
        suggested_button = ctk.CTkButton(target_frame, text="Use Suggested English Lyrics", height=42, command=lambda: self.set_target_text(SUGGESTED_TARGET_LYRICS))
        suggested_button.grid(row=5, column=0, padx=14, pady=6, sticky="ew")
        generate_button = ctk.CTkButton(target_frame, text="Generate Edited Song", height=52, font=ctk.CTkFont(size=16, weight="bold"), command=self.start_generation)
        generate_button.grid(row=6, column=0, padx=14, pady=(8, 6), sticky="ew")
        generation_help = ctk.CTkLabel(target_frame, text="Generation only starts when token counts and every pause position match.", anchor="w", wraplength=430, justify="left")
        generation_help.grid(row=7, column=0, padx=14, pady=(0, 14), sticky="w")
        log_frame = ctk.CTkFrame(right_scroll)
        log_frame.grid(row=1, column=0, padx=8, pady=8, sticky="ew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_title = ctk.CTkLabel(log_frame, text="Application Log", font=ctk.CTkFont(size=18, weight="bold"))
        log_title.grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")
        log_help = ctk.CTkLabel(log_frame, text="Validation details, token conflicts, preprocessing messages, and generation errors appear here.", anchor="w", wraplength=430, justify="left")
        log_help.grid(row=1, column=0, padx=14, pady=(0, 6), sticky="w")
        self.log_box = ctk.CTkTextbox(log_frame, height=360, wrap="word")
        self.log_box.grid(row=2, column=0, padx=14, pady=6, sticky="ew")
        progress = ctk.CTkProgressBar(log_frame, variable=self.progress_value)
        progress.grid(row=3, column=0, padx=14, pady=(8, 14), sticky="ew")

    def verify_cuda(self):
        numpy_major = int(np.__version__.split(".")[0])
        if numpy_major >= 2:
            raise RuntimeError(f"NumPy {np.__version__} is incompatible with this SoulX-Singer PyTorch environment. Install NumPy 1.26.4.")
        if not torch.cuda.is_available():
            torch_version = torch.__version__
            cuda_version = torch.version.cuda
            if cuda_version is None:
                raise RuntimeError(f"PyTorch {torch_version} is CPU-only or was installed without CUDA support.")
            raise RuntimeError(f"PyTorch {torch_version} reports CUDA {cuda_version}, but CUDA is not available.")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.write_log(f"CUDA available: {torch.cuda.get_device_name(0)}")
        self.write_log(f"PyTorch version: {torch.__version__}")
        self.write_log(f"PyTorch CUDA version: {torch.version.cuda}")
        self.write_log(f"NumPy version: {np.__version__}")
        self.write_log(f"CUDA device count: {torch.cuda.device_count()}")
        self.write_log(f"FFmpeg path: {FFMPEG_PATH}")
        self.write_log(f"Parakeet model: {PARAKEET_MODEL_PATH}")

    def cleanup_gpu(self):
        self.write_log("Cleaning GPU memory.")
        if self.svs_model is not None:
            try:
                del self.svs_model
            except Exception:
                pass
            self.svs_model = None
        if self.preprocess_pipeline is not None:
            try:
                del self.preprocess_pipeline
            except Exception:
                pass
            self.preprocess_pipeline = None
        self.svs_config = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
            self.write_log(f"GPU memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved.")

    def browse_song(self):
        if self.is_busy:
            self.write_log("Cannot change input while processing is running.")
            return
        path = filedialog.askopenfilename(title="Select Audio or Video", filetypes=[("Audio and Video", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.wma *.mp4 *.mkv *.mov *.avi *.webm *.wmv *.m4v"), ("Audio Files", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.wma"), ("Video Files", "*.mp4 *.mkv *.mov *.avi *.webm *.wmv *.m4v"), ("All Files", "*.*")])
        if not path:
            return
        self.stop_all_playback()
        self.song_path.set(path)
        self.converted_audio_path = None
        self.vocal_path = None
        self.instrumental_path = None
        self.original_metadata_path = None
        self.target_metadata_path = None
        self.prompt_metadata_path = None
        self.session_dir = None
        self.generated_final_path = None
        self.original_text.configure(state="normal")
        self.original_text.delete("1.0", "end")
        self.original_text.configure(state="disabled")
        self.target_text.delete("1.0", "end")
        self.original_token_layout = None
        self.update_token_counters()
        self.write_log(f"Selected input: {path}")
        threading.Thread(target=self.auto_prepare_and_transcribe, args=(path,), daemon=True).start()

    def browse_output(self):
        path = filedialog.asksaveasfilename(title="Save Edited Song", defaultextension=".wav", filetypes=[("WAV Audio", "*.wav"), ("All Files", "*.*")])
        if not path:
            return
        destination = Path(path).expanduser().resolve()
        if destination.suffix.lower() != ".wav":
            destination = destination.with_suffix(".wav")
        self.output_path.set(str(destination))
        if self.generated_final_path is not None and self.generated_final_path.is_file():
            try:
                saved_path = self.save_output_file(self.generated_final_path, destination)
                self.write_log(f"Saved existing generated WAV: {saved_path}")
            except Exception as exc:
                self.write_log(f"Save As failed: {exc}")
                self.write_exception("SAVE AS ERROR", exc)
            return
        self.write_log(f"Output path set for next generation: {destination}")

    def write_log(self, text):
        print(text, flush=True)
        self.after(0, self._write_log, text)

    def _write_log(self, text):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.status.set(text)

    def write_exception(self, title, exc):
        print("", flush=True)
        print("=" * 100, flush=True)
        print(title, flush=True)
        print("=" * 100, flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        print("=" * 100, flush=True)

    def set_progress(self, value):
        self.after(0, self.progress_value.set, value)

    def ensure_ffmpeg(self):
        try:
            result = subprocess.run([str(FFMPEG_PATH), "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.returncode == 0
        except Exception:
            traceback.print_exc()
            return False

    def convert_to_wav(self, input_path):
        if Path(input_path).suffix.lower() == ".wav":
            self.converted_audio_path = input_path
            return input_path
        if not self.ensure_ffmpeg():
            raise RuntimeError(f"FFmpeg could not be executed from {FFMPEG_PATH}")
        output_dir = Path(tempfile.gettempdir()) / "soulxsinger_gui"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(input_path).stem}_converted.wav"
        command = [str(FFMPEG_PATH), "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(output_path)]
        self.write_log("Converting input to WAV with FFmpeg.")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed:\n{result.stderr[-4000:]}")
        if not output_path.is_file():
            raise RuntimeError("FFmpeg did not create the WAV file.")
        self.converted_audio_path = str(output_path)
        self.write_log(f"WAV ready: {self.converted_audio_path}")
        return self.converted_audio_path

    def create_session(self):
        root = Path(SOULXSINGER_ROOT) / "outputs" / "soulxsinger_gui"
        root.mkdir(parents=True, exist_ok=True)
        session = root / next(tempfile._get_candidate_names())
        (session / "audio").mkdir(parents=True, exist_ok=True)
        (session / "transcriptions" / "prompt").mkdir(parents=True, exist_ok=True)
        (session / "transcriptions" / "target").mkdir(parents=True, exist_ok=True)
        (session / "generated").mkdir(parents=True, exist_ok=True)
        self.session_dir = session
        return session

    def get_soul_language(self):
        value = self.language.get()
        if value not in ("English", "Mandarin", "Cantonese"):
            return "English"
        return value

    def load_preprocess_pipeline(self):
        if self.preprocess_pipeline is not None:
            return self.preprocess_pipeline
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SoulX-Singer preprocessing.")
        if self.get_soul_language() == "English" and not PARAKEET_MODEL_PATH.is_file():
            raise FileNotFoundError(f"SoulX-Singer Parakeet English ASR model was not found at {PARAKEET_MODEL_PATH}")
        self.write_log("Loading SoulX-Singer preprocessing pipeline on CUDA:0.")
        self.write_log(f"SoulX-Singer English ASR model: {PARAKEET_MODEL_PATH}")
        self.preprocess_pipeline = PreprocessPipeline(device="cuda:0", language=self.get_soul_language(), save_dir=str(Path(SOULXSINGER_ROOT) / "outputs" / "soulxsinger_gui" / "preprocess"), vocal_sep=True, max_merge_duration=60000, midi_transcribe=True)
        return self.preprocess_pipeline

    def run_preprocess(self, audio_path, save_path):
        pipeline = self.load_preprocess_pipeline()
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        pipeline.save_dir = str(save_path)
        pipeline.language = self.get_soul_language()
        pipeline.midi_transcribe = True
        if pipeline.vocal_detector is not None:
            pipeline.vocal_detector.cut_wavs_output_dir = str(save_path / "cut_wavs")
        self.write_log("Separating vocal and accompaniment first, then processing the separated vocal for F0, lyrics, and notes.")
        pipeline.run(audio_path=str(audio_path), vocal_sep=True, max_merge_duration=60000, language=self.get_soul_language())
        metadata_path = save_path / "metadata.json"
        vocal_path = save_path / "vocal.wav"
        accompaniment_path = save_path / "acc.wav"
        if not metadata_path.is_file():
            raise RuntimeError(f"SoulX-Singer preprocessing did not create {metadata_path}")
        if not vocal_path.is_file():
            raise RuntimeError(f"SoulX-Singer preprocessing did not create {vocal_path}")
        if not accompaniment_path.is_file():
            raise RuntimeError(f"SoulX-Singer preprocessing did not create {accompaniment_path}")
        self.write_log(f"Separated vocal ready: {vocal_path}")
        self.write_log(f"Separated accompaniment ready: {accompaniment_path}")
        self.write_log(f"SoulX metadata ready: {metadata_path}")
        return metadata_path, vocal_path, accompaniment_path

    def transcribe_for_display(self, metadata_path):
        transcript = self.read_metadata_text(metadata_path)
        detected_language = self.get_soul_language()
        if not transcript:
            transcript = ""
        return transcript, detected_language

    def read_metadata_text(self, metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        phrases = []
        for segment in metadata:
            text = " ".join(str(segment.get("text", "")).split())
            if text:
                phrases.append(text)
        return "|".join(phrases)

    def copy_prompt_metadata(self, source_metadata):
        prompt_path = self.session_dir / "transcriptions" / "prompt" / "metadata.json"
        shutil.copy2(source_metadata, prompt_path)
        self.prompt_metadata_path = prompt_path
        return prompt_path

    def validate_token_sequence(self, source_sequence, target_sequence):
        if len(source_sequence) != len(target_sequence):
            raise RuntimeError(f"New Lyrics contains {len(target_sequence)} total tokens including pause markers, but SoulX detected {len(source_sequence)} total metadata tokens. Keep singing-token count and <SP>/<AP> placement aligned with the original metadata.")
        source_singing_count = len([token for token in source_sequence if token not in ("<SP>", "<AP>")])
        target_singing_count = len([token for token in target_sequence if token not in ("<SP>", "<AP>")])
        if source_singing_count != target_singing_count:
            raise RuntimeError(f"New Lyrics contains {target_singing_count} singing tokens, but SoulX detected {source_singing_count} singing tokens.")
        layout_matches, conflict = self.compare_token_layout(source_sequence, target_sequence)
        if not layout_matches and conflict is not None:
            index, source_token, target_token = conflict
            raise RuntimeError(f"Pause-marker conflict at metadata token {index}: the original has {source_token}, but the new lyrics has {target_token}. Keep <SP>/<AP> in the same positions as the original lyrics.")

    def make_target_metadata(self, source_metadata, target_text):
        with open(source_metadata, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        source_sequence = []
        for segment in metadata:
            source_sequence.extend([token for token in str(segment.get("text", "")).split() if token])
        target_sequence = self.tokenize_target_lyrics(target_text)
        self.validate_token_sequence(source_sequence, target_sequence)
        singing_count = len([token for token in target_sequence if token not in ("<SP>", "<AP>")])
        cursor = 0
        language = self.get_soul_language()
        for segment in metadata:
            old_tokens = str(segment.get("text", "")).split()
            replaced_tokens = []
            phoneme_tokens = []
            for old_token in old_tokens:
                target_token = target_sequence[cursor]
                if old_token in ("<SP>", "<AP>"):
                    replaced_tokens.append(target_token)
                    phoneme_tokens.append(target_token)
                else:
                    new_token = target_token
                    replaced_tokens.append(new_token)
                    generated_phoneme = g2p_transform([new_token], language)[0]
                    if not generated_phoneme:
                        raise RuntimeError(f"Could not generate a phoneme for lyric token: {new_token}")
                    phoneme_tokens.append(generated_phoneme)
                cursor += 1
            segment["text"] = " ".join(replaced_tokens)
            segment["phoneme"] = " ".join(phoneme_tokens)
        target_path = self.session_dir / "transcriptions" / "target" / "edited_metadata.json"
        with open(target_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        self.target_metadata_path = target_path
        self.write_log(f"Target metadata tokens: {len(target_sequence)}")
        self.write_log(f"Target singing tokens: {singing_count}")
        self.write_log("Target metadata preserves <SP>/<AP> positions exactly and regenerates phonemes.")
        return target_path

    def tokenize_target_lyrics(self, text):
        normalized = text.replace("\r", "\n").replace("|", " ")
        return [token.strip() for token in normalized.split() if token.strip()]

    def build_svs_model(self):
        if self.svs_model is not None:
            return self.svs_model
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SoulX-Singer inference.")
        model_path = Path(SOULXSINGER_ROOT) / "pretrained_models" / "SoulX-Singer" / "model.pt"
        config_path = Path(SOULXSINGER_ROOT) / "soulxsinger" / "config" / "soulxsinger.yaml"
        phoneset_path = Path(SOULXSINGER_ROOT) / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
        if not model_path.is_file():
            raise FileNotFoundError(f"SoulX-Singer model was not found at {model_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"SoulX-Singer config was not found at {config_path}")
        if not phoneset_path.is_file():
            raise FileNotFoundError(f"SoulX-Singer phoneset was not found at {phoneset_path}")
        self.write_log(f"Loading SoulX-Singer model from {model_path}")
        self.svs_config = load_config(str(config_path))
        self.svs_model = build_svs_model(model_path=str(model_path), config=self.svs_config, device="cuda:0", use_fp16=bool(self.fp16.get()))
        self.write_log("SoulX-Singer model loaded.")
        return self.svs_model

    def run_soul_inference(self, output_directory):
        model = self.build_svs_model()
        config = self.svs_config
        phoneset_path = Path(SOULXSINGER_ROOT) / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"

        class Args:
            pass

        args = Args()
        args.device = "cuda:0"
        args.model_path = str(Path(SOULXSINGER_ROOT) / "pretrained_models" / "SoulX-Singer" / "model.pt")
        args.config = str(Path(SOULXSINGER_ROOT) / "soulxsinger" / "config" / "soulxsinger.yaml")
        args.prompt_wav_path = str(self.vocal_path)
        args.prompt_metadata_path = str(self.prompt_metadata_path)
        args.target_metadata_path = str(self.target_metadata_path)
        args.phoneset_path = str(phoneset_path)
        args.save_dir = str(output_directory)
        args.auto_shift = bool(self.auto_shift.get())
        args.pitch_shift = int(self.pitch_shift.get())
        args.control = self.control.get()
        args.use_fp16 = bool(self.fp16.get()) and torch.cuda.is_available()
        self.write_log(f"SoulX control mode: {args.control}")
        self.write_log(f"SoulX auto pitch shift: {args.auto_shift}")
        self.write_log(f"SoulX pitch shift: {args.pitch_shift}")
        self.write_log("Starting official SoulX-Singer SVS inference.")
        svs_process(args, config, model)
        generated = Path(output_directory) / "generated.wav"
        if not generated.is_file():
            raise RuntimeError(f"SoulX-Singer finished without creating {generated}")
        return generated

    def save_output_file(self, source_path, destination_path):
        source = Path(source_path).expanduser().resolve()
        destination = Path(destination_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Generated source WAV was not found: {source}")
        if source == destination:
            if source.stat().st_size <= 0:
                raise RuntimeError(f"Generated WAV is empty: {source}")
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(f"{destination.stem}.saving.wav")
        if temp_path.exists():
            temp_path.unlink()
        shutil.copy2(source, temp_path)
        if not temp_path.is_file() or temp_path.stat().st_size <= 0:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Temporary WAV save failed: {temp_path}")
        temp_path.replace(destination)
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise RuntimeError(f"Saved WAV could not be verified: {destination}")
        return destination

    def mix_with_accompaniment(self, generated_path, accompaniment_path, output_path):
        generated, _ = librosa.load(str(generated_path), sr=24000, mono=True)
        accompaniment, _ = librosa.load(str(accompaniment_path), sr=24000, mono=True)
        length = min(len(generated), len(accompaniment))
        if length <= 0:
            raise RuntimeError("Generated vocal or accompaniment is empty.")
        mixed = generated[:length] + accompaniment[:length]
        peak = float(np.max(np.abs(mixed))) if mixed.size else 1.0
        if peak > 1.0:
            mixed = mixed / peak
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f"{output_path.stem}.mixing.wav")
        if temp_path.exists():
            temp_path.unlink()
        sf.write(str(temp_path), mixed, 24000)
        if not temp_path.is_file() or temp_path.stat().st_size <= 0:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Temporary mixed WAV was not created: {temp_path}")
        temp_path.replace(output_path)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"Mixed WAV could not be verified: {output_path}")
        return output_path

    def auto_prepare_and_transcribe(self, song):
        if self.is_busy:
            return
        self.is_busy = True
        try:
            self.set_progress(0.03)
            wav_path = self.convert_to_wav(song)
            self.set_progress(0.10)
            self.session_dir = self.create_session()
            target_dir = self.session_dir / "transcriptions" / "target"
            self.write_log(f"Preparing separated vocal, accompaniment, and SoulX metadata in {target_dir}")
            self.original_metadata_path, self.vocal_path, accompaniment_path = self.run_preprocess(wav_path, target_dir)
            self.instrumental_path = str(accompaniment_path)
            self.prompt_metadata_path = self.copy_prompt_metadata(self.original_metadata_path)
            self.set_progress(0.55)
            self.write_log("Reading original lyrics from SoulX-Singer metadata.")
            transcript, detected_language = self.transcribe_for_display(self.original_metadata_path)
            source_tokens = self.tokenize_target_lyrics(transcript)
            source_singing_count = len([token for token in source_tokens if token not in ("<SP>", "<AP>")])
            source_sp_count = len([token for token in source_tokens if token in ("<SP>", "<AP>")])
            self.set_progress(0.80)
            self.after(0, self.set_original_text, transcript)
            self.set_progress(1.0)
            self.write_log(f"Detected language: {detected_language}")
            self.write_log(f"Detected metadata singing tokens: {source_singing_count}")
            self.write_log(f"Detected metadata pause markers: {source_sp_count}")
            self.write_log(f"Separated vocal used for SoulX processing: {self.vocal_path}")
            self.write_log(f"Accompaniment reserved for final mix: {self.instrumental_path}")
            self.write_log("Original lyrics are displayed.")
            if source_singing_count == 33 and source_sp_count == 8 and len(source_tokens) == 41:
                self.write_log("Suggested English New Lyrics were loaded with the same token and pause layout.")
            self.write_log("Enter New Lyrics using the same singing-token count and the same <SP>/<AP> positions as the original metadata.")
        except torch.cuda.OutOfMemoryError as exc:
            self.cleanup_gpu()
            self.set_progress(0.0)
            self.write_log("CUDA ran out of VRAM during SoulX preprocessing.")
            self.write_exception("CUDA OUT OF MEMORY", exc)
        except Exception as exc:
            self.cleanup_gpu()
            self.set_progress(0.0)
            self.write_log(f"Automatic SoulX preprocessing failed: {exc}")
            self.write_exception("AUTOMATIC PREPROCESSING ERROR", exc)
        finally:
            self.is_busy = False

    def set_original_text(self, text):
        self.original_text.configure(state="normal")
        self.original_text.delete("1.0", "end")
        self.original_text.insert("1.0", text)
        self.original_text.configure(state="disabled")
        self.original_token_layout = self.tokenize_target_lyrics(text)
        source_singing, source_pauses, source_total = self.count_lyrics_tokens(text)
        if not self.target_text.get("1.0", "end").strip() and source_singing == 33 and source_pauses == 8 and source_total == 41:
            self.set_target_text(SUGGESTED_TARGET_LYRICS)
        self.update_token_counters()

    def set_target_text(self, text):
        self.target_text.delete("1.0", "end")
        self.target_text.insert("1.0", text)
        self.update_token_counters()

    def count_lyrics_tokens(self, text):
        tokens = self.tokenize_target_lyrics(text)
        singing_count = len([token for token in tokens if token not in ("<SP>", "<AP>")])
        pause_count = len([token for token in tokens if token in ("<SP>", "<AP>")])
        return singing_count, pause_count, len(tokens)

    def get_pause_layout(self, tokens):
        return [(index + 1, token) for index, token in enumerate(tokens) if token in ("<SP>", "<AP>")]

    def get_pause_layout_text(self, tokens):
        layout = self.get_pause_layout(tokens)
        if not layout:
            return "-"
        return ",".join(f"{index}:{token}" for index, token in layout)

    def compare_token_layout(self, source_tokens, target_tokens):
        if len(source_tokens) != len(target_tokens):
            return False, None
        for index, source_token in enumerate(source_tokens):
            target_token = target_tokens[index]
            source_is_pause = source_token in ("<SP>", "<AP>")
            target_is_pause = target_token in ("<SP>", "<AP>")
            if source_is_pause != target_is_pause or (source_is_pause and source_token != target_token):
                return False, (index + 1, source_token, target_token)
        return True, None

    def validate_current_lyrics(self):
        original = self.get_original_lyrics()
        target = self.get_target_lyrics()
        source_tokens = self.original_token_layout[:] if self.original_token_layout is not None else self.tokenize_target_lyrics(original)
        target_tokens = self.tokenize_target_lyrics(target)
        source_singing, source_pauses, source_total = self.count_lyrics_tokens(original)
        target_singing, target_pauses, target_total = self.count_lyrics_tokens(target)
        source_layout = self.get_pause_layout_text(source_tokens)
        target_layout = self.get_pause_layout_text(target_tokens)
        lines = ["", "===== SOULX LYRIC VALIDATION =====", f"Original singing tokens: {source_singing}", f"Original pause markers: {source_pauses}", f"Original total tokens: {source_total}", f"New singing tokens: {target_singing}", f"New pause markers: {target_pauses}", f"New total tokens: {target_total}", f"Original pause positions: {source_layout}", f"New pause positions: {target_layout}"]
        matched = source_total == target_total and source_singing == target_singing and source_pauses == target_pauses
        if matched:
            layout_matches, conflict = self.compare_token_layout(source_tokens, target_tokens)
            matched = layout_matches and conflict is None
        if matched:
            lines.append("RESULT: MATCH")
            lines.append("All singing-token counts, pause counts, total tokens, and <SP>/<AP> positions match.")
        else:
            lines.append("RESULT: MISMATCH")
            if source_total != target_total:
                lines.append(f"Total token mismatch: original={source_total}, new={target_total}.")
            if source_singing != target_singing:
                lines.append(f"Singing-token mismatch: original={source_singing}, new={target_singing}.")
            if source_pauses != target_pauses:
                lines.append(f"Pause-marker count mismatch: original={source_pauses}, new={target_pauses}.")
            conflicts = []
            for index in range(max(len(source_tokens), len(target_tokens))):
                source_token = source_tokens[index] if index < len(source_tokens) else "<MISSING>"
                target_token = target_tokens[index] if index < len(target_tokens) else "<MISSING>"
                source_is_pause = source_token in ("<SP>", "<AP>")
                target_is_pause = target_token in ("<SP>", "<AP>")
                if source_is_pause != target_is_pause or (source_is_pause and source_token != target_token):
                    conflicts.append(f"Token {index + 1}: original={source_token} | new={target_token}")
            if conflicts:
                lines.append("Pause-position conflicts:")
                lines.extend(conflicts)
            else:
                lines.append("No individual pause-position conflict was found; the mismatch is in counts or total length.")
        for line in lines:
            self.write_log(line)
        self.update_token_counters()
        return matched

    def update_token_counters(self, event=None):
        original = self.get_original_lyrics()
        target = self.get_target_lyrics()
        original_tokens = self.original_token_layout[:] if self.original_token_layout is not None else self.tokenize_target_lyrics(original)
        target_tokens = self.tokenize_target_lyrics(target)
        original_singing, original_pauses, original_total = self.count_lyrics_tokens(original)
        target_singing, target_pauses, target_total = self.count_lyrics_tokens(target)
        layout_matches, conflict = self.compare_token_layout(original_tokens, target_tokens)
        count_matches = target_singing == original_singing and target_pauses == original_pauses and target_total == original_total
        target_status = "MATCH" if count_matches and layout_matches and conflict is None else "MISMATCH"
        original_layout_text = self.get_pause_layout_text(original_tokens)
        target_layout_text = self.get_pause_layout_text(target_tokens)
        self.original_token_counter.set(f"Singing: {original_singing} | Pauses: {original_pauses} | Total: {original_total} | Pause positions: {original_layout_text}")
        self.target_token_counter.set(f"Singing: {target_singing} | Pauses: {target_pauses} | Total: {target_total} | Pause positions: {target_layout_text} | Required: {original_singing}/{original_pauses}/{original_total} | {target_status}")

    def normalize_lyrics(self, text):
        lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
        merged = []
        for line in lines:
            parts = [part.strip() for part in line.split("|") if part.strip()]
            merged.extend(parts)
        return "|".join(merged)

    def get_original_lyrics(self):
        return self.normalize_lyrics(self.original_text.get("1.0", "end").strip())

    def get_target_lyrics(self):
        return self.normalize_lyrics(self.target_text.get("1.0", "end").strip())

    def start_generation(self):
        if self.is_busy:
            self.write_log("Please wait until preprocessing finishes.")
            return
        original = self.get_original_lyrics()
        target = self.get_target_lyrics()
        if not self.song_path.get().strip():
            self.write_log("No input file selected.")
            return
        if not original:
            self.write_log("Original lyrics are empty.")
            return
        if not target:
            self.write_log("New lyrics are empty.")
            return
        if self.original_metadata_path is None or self.vocal_path is None:
            self.write_log("SoulX preprocessing is not ready.")
            return
        if not self.validate_current_lyrics():
            self.write_log("Generation cancelled because lyric validation returned MISMATCH.")
            return
        self.is_busy = True
        self.stop_all_playback()
        threading.Thread(target=self.generate_worker, args=(original, target), daemon=True).start()

    def generate_worker(self, original, target):
        try:
            self.set_progress(0.05)
            if self.vocal_path is None or self.original_metadata_path is None or self.prompt_metadata_path is None:
                raise RuntimeError("SoulX preprocessing is not ready.")
            output = self.output_path.get().strip()
            if not output:
                raise RuntimeError("Output path is empty.")
            output_path = Path(output).expanduser().resolve()
            if output_path.suffix.lower() != ".wav":
                output_path = output_path.with_suffix(".wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.set(str(output_path))
            self.set_progress(0.15)
            self.write_log("Applying New Lyrics to SoulX target metadata.")
            self.target_metadata_path = self.make_target_metadata(self.original_metadata_path, target)
            self.set_progress(0.30)
            generated_dir = self.session_dir / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            generated_vocal = self.run_soul_inference(generated_dir)
            self.cleanup_model_only()
            self.set_progress(0.82)
            master_path = generated_dir / "edited_song.wav"
            if self.instrumental_path and os.path.isfile(self.instrumental_path):
                self.write_log("Mixing generated SoulX vocal with the separated accompaniment.")
                self.mix_with_accompaniment(generated_vocal, self.instrumental_path, master_path)
            else:
                self.save_output_file(generated_vocal, master_path)
            self.generated_final_path = master_path.resolve()
            saved_path = self.save_output_file(self.generated_final_path, output_path)
            if not saved_path.is_file():
                raise RuntimeError(f"SoulX-Singer finished without creating the requested output WAV: {saved_path}")
            self.set_progress(1.0)
            self.write_log(f"Master WAV ready: {self.generated_final_path}")
            self.write_log(f"Saved WAV ready: {saved_path}")
            self.write_log(f"Saved WAV size: {saved_path.stat().st_size:,} bytes")
        except torch.cuda.OutOfMemoryError as exc:
            self.cleanup_gpu()
            self.set_progress(0.0)
            self.write_log("CUDA ran out of VRAM during SoulX-Singer generation.")
            self.write_exception("SOULX CUDA OUT OF MEMORY", exc)
        except Exception as exc:
            self.cleanup_gpu()
            self.set_progress(0.0)
            self.write_log(f"Generation failed: {exc}")
            self.write_exception("SOULX GENERATION ERROR", exc)
        finally:
            self.is_busy = False

    def cleanup_model_only(self):
        if self.svs_model is not None:
            try:
                del self.svs_model
            except Exception:
                pass
            self.svs_model = None
        self.svs_config = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def toggle_source_playback(self):
        if self.previewing_source:
            self.stop_source_playback()
            return
        path = self.song_path.get().strip()
        if not path or not os.path.isfile(path):
            self.write_log("No valid source file selected.")
            return
        threading.Thread(target=self.start_source_playback, args=(path,), daemon=True).start()

    def start_source_playback(self, path):
        try:
            audio_path = self.convert_to_wav(path)
            self.stop_output_playback()
            self.previewing_source = True
            self.after(0, self.source_play_button.configure, {"text": "■ Stop"})
            self.write_log("Playing source preview.")
            winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except Exception as exc:
            self.previewing_source = False
            self.after(0, self.source_play_button.configure, {"text": "▶ Play"})
            self.write_log(f"Source preview failed: {exc}")
            self.write_exception("SOURCE PREVIEW ERROR", exc)

    def toggle_output_playback(self):
        if self.previewing_output:
            self.stop_output_playback()
            return
        output = self.output_path.get().strip()
        if not output or not os.path.isfile(output):
            if self.generated_final_path is None or not self.generated_final_path.is_file():
                self.write_log("Output file does not exist yet.")
                return
            output = str(self.generated_final_path)
        self.start_output_playback(output)

    def start_output_playback(self, path):
        try:
            self.stop_source_playback()
            self.previewing_output = True
            self.output_play_button.configure(text="■ Stop")
            self.write_log("Playing output preview.")
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except Exception as exc:
            self.previewing_output = False
            self.output_play_button.configure(text="▶ Play")
            self.write_log(f"Output preview failed: {exc}")
            self.write_exception("OUTPUT PREVIEW ERROR", exc)

    def stop_source_playback(self):
        winsound.PlaySound(None, winsound.SND_PURGE)
        self.previewing_source = False
        self.source_play_button.configure(text="▶ Play")

    def stop_output_playback(self):
        winsound.PlaySound(None, winsound.SND_PURGE)
        self.previewing_output = False
        self.output_play_button.configure(text="▶ Play")

    def stop_all_playback(self):
        winsound.PlaySound(None, winsound.SND_PURGE)
        self.previewing_source = False
        self.previewing_output = False
        self.source_play_button.configure(text="▶ Play")
        self.output_play_button.configure(text="▶ Play")

    def close_app(self):
        self.stop_all_playback()
        self.cleanup_gpu()
        self.destroy()

if __name__ == "__main__":
    app = SoulXSingerLyricsGUI()
    app.mainloop()