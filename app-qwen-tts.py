import os
import gc
import threading
import traceback
import time
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import subprocess
import tempfile
import customtkinter as ctk
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import BitsAndBytesConfig
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".qwen3_tts_cache")
MDX_MODEL = "UVR-MDX-NET-Voc_FT.onnx"
MDX_MODEL_DIR = os.path.join(CACHE_DIR, "mdx_models")
MDX_OUTPUT_DIR = os.path.join(CACHE_DIR, "mdx_output")
WHISPER_MODEL = "small"
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]
REFERENCE_EXTENSIONS = "*.wav *.flac *.ogg *.mp3 *.m4a *.aac *.wma *.mp4 *.mkv *.avi *.mov *.webm *.wmv *.mpeg *.mpg *.m4v *.3gp *.ts *.mts *.m2ts *.flv *.f4v *.vob *.ogv"
REFERENCE_SAMPLE_RATE = 24000
SEPARATION_SAMPLE_RATE = 44100
REFERENCE_TARGET_RMS_DB = -20.0
REFERENCE_MAX_PEAK_DB = -3.0
REFERENCE_MIN_SECONDS = 3.0
REFERENCE_MAX_SECONDS = 30.0
SILENCE_FRAME_MS = 20
SILENCE_HOP_MS = 10
SILENCE_THRESHOLD_DB = -48.0
SILENCE_RELATIVE_THRESHOLD_DB = 30.0
SILENCE_NOISE_MARGIN_DB = 3.0
SILENCE_SPLIT_GAP_SECONDS = 0.30
MAX_INTERNAL_SILENCE_SECONDS = 0.30
MIN_SPEECH_SECONDS = 0.08
LEADING_TRAILING_PADDING_SECONDS = 0.06
MAX_SINGLE_GENERATION_CHARS = 320
SEGMENT_MAX_CHARS = 180
SEGMENT_MIN_CHARS = 45
SEGMENT_JOIN_PAUSE_SECONDS = 0.04
MAX_BATCH_SIZE = 3
GENERATION_REPETITION_PENALTY = 1.05
GENERATION_SINGLE_MAX_NEW_TOKENS = 2048
GENERATION_SEGMENT_MAX_NEW_TOKENS = 1024
MDX_SEGMENT_SIZE = 256
MDX_OVERLAP = 0.25
MDX_BATCH_SIZE = 1
MDX_HOP_LENGTH = 1024
MDX_ENABLE_DENOISE = True

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MDX_MODEL_DIR, exist_ok=True)
os.makedirs(MDX_OUTPUT_DIR, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme(str(BASE_DIR / "themes" / "red.json"))

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

class QwenTTSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Qwen3-TTS 1.7B Maximum Stability ICL Voice Clone")
        self.geometry("1100x790")
        self.minsize(930, 700)
        self.model = None
        self.reference_path = None
        self.clean_reference_audio = None
        self.clean_reference_sr = None
        self.generated_audio = None
        self.generated_sr = None
        self.voice_prompt = None
        self.voice_prompt_key = None
        self.enhancer = None
        self.enhancer_device = None
        self.whisper_model = None
        self.loading = False
        self.processing_reference = False
        self.transcribing_reference = False
        self.generating = False
        self.generate_start_time = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.create_header()
        self.create_controls()
        self.create_text_area()
        self.create_bottom()
        self.after(100, self.maximize_window)
        self.after(300, self.load_model_thread)

    def maximize_window(self):
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                pass

    def create_header(self):
        frame = ctk.CTkFrame(self, corner_radius=0)
        frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        frame.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(frame, text="Qwen3-TTS 1.7B Maximum Stability ICL Voice Clone", font=ctk.CTkFont(size=26, weight="bold"))
        title.grid(row=0, column=0, padx=25, pady=(20, 20), sticky="w")

    def create_controls(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="ew", padx=20, pady=20)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(4, weight=1)
        self.ref_button = ctk.CTkButton(frame, text="Select Voice Sample", command=self.select_reference)
        self.ref_button.grid(row=0, column=0, padx=(15, 10), pady=15)
        self.ref_label = ctk.CTkLabel(frame, text="No reference audio selected", anchor="w")
        self.ref_label.grid(row=0, column=1, padx=10, pady=15, sticky="ew")
        self.check_button = ctk.CTkButton(frame, text="Check Cleaned Voice", command=self.check_voice_sample, state="disabled")
        self.check_button.grid(row=0, column=2, padx=(10, 10), pady=15, sticky="ew")
        self.reload_button = ctk.CTkButton(frame, text="Reload Reference Transcript", command=self.reload_reference_transcript, state="disabled")
        self.reload_button.grid(row=0, column=3, padx=10, pady=15, sticky="ew")
        self.device_label = ctk.CTkLabel(frame, text=self.get_device_text(), anchor="w")
        self.device_label.grid(row=0, column=4, padx=(10, 15), pady=15, sticky="ew")
        self.language_label = ctk.CTkLabel(frame, text="Language")
        self.language_label.grid(row=1, column=0, padx=(15, 10), pady=(5, 15), sticky="w")
        self.language_menu = ctk.CTkOptionMenu(frame, values=LANGUAGES)
        self.language_menu.set("English")
        self.language_menu.grid(row=1, column=1, padx=10, pady=(5, 15), sticky="ew")
        self.mode_label = ctk.CTkLabel(frame, text="ICL Voice Clone: ON")
        self.mode_label.grid(row=1, column=2, padx=10, pady=(5, 15))
        self.stability_label = ctk.CTkLabel(frame, text="Maximum Stability: ON")
        self.stability_label.grid(row=1, column=3, padx=10, pady=(5, 15))
        self.clear_button = ctk.CTkButton(frame, text="Clear", command=self.clear_all, fg_color="gray35", hover_color="gray25")
        self.clear_button.grid(row=1, column=4, padx=(10, 15), pady=(5, 15), sticky="ew")

    def create_text_area(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(4, weight=1)
        text_title = ctk.CTkLabel(frame, text="Text to Generate", font=ctk.CTkFont(size=17, weight="bold"))
        text_title.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        self.text_box = ctk.CTkTextbox(frame, wrap="word", font=ctk.CTkFont(size=16))
        self.text_box.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.ref_title = ctk.CTkLabel(frame, text="Reference Transcript", font=ctk.CTkFont(size=17, weight="bold"))
        self.ref_title.grid(row=2, column=0, padx=15, pady=(0, 8), sticky="w")
        self.ref_info = ctk.CTkLabel(frame, text="Verify every word against the cleaned reference. Internal pauses are preserved exactly.", font=ctk.CTkFont(size=12), text_color="gray70")
        self.ref_info.grid(row=3, column=0, padx=15, pady=(0, 8), sticky="w")
        self.ref_text_box = ctk.CTkTextbox(frame, height=110, wrap="word", font=ctk.CTkFont(size=15))
        self.ref_text_box.grid(row=4, column=0, padx=15, pady=(0, 15), sticky="nsew")

    def create_bottom(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 20))
        frame.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(frame, mode="indeterminate")
        self.progress.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="ew")
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="ew")
        button_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.generate_button = ctk.CTkButton(button_frame, text="Generate Speech", command=self.generate_thread, height=42, font=ctk.CTkFont(size=15, weight="bold"))
        self.generate_button.grid(row=0, column=0, padx=5, sticky="ew")
        self.play_button = ctk.CTkButton(button_frame, text="Play", command=self.play_audio, height=42, state="disabled")
        self.play_button.grid(row=0, column=1, padx=5, sticky="ew")
        self.save_button = ctk.CTkButton(button_frame, text="Save WAV", command=self.save_audio, height=42, state="disabled")
        self.save_button.grid(row=0, column=2, padx=5, sticky="ew")
        self.download_reference_button = ctk.CTkButton(button_frame, text="Download Voice Sample", command=self.download_voice_sample, height=42, state="disabled")
        self.download_reference_button.grid(row=0, column=3, padx=5, sticky="ew")
        self.footer = ctk.CTkLabel(frame, text="Loading Qwen3-TTS...", font=ctk.CTkFont(size=12), text_color="gray70")
        self.footer.grid(row=2, column=0, padx=15, pady=(0, 12), sticky="w")

    def get_device_text(self):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"GPU: {name} | {vram:.1f} GB"
        return "GPU: CUDA not detected"

    def select_reference(self):
        if self.processing_reference or self.transcribing_reference or self.generating or self.loading:
            return
        path = filedialog.askopenfilename(title="Select reference voice or video", filetypes=[("Audio and Video Files", REFERENCE_EXTENSIONS), ("All Files", "*.*")])
        if not path:
            return
        self.reference_path = path
        self.clean_reference_audio = None
        self.clean_reference_sr = None
        self.generated_audio = None
        self.generated_sr = None
        self.voice_prompt = None
        self.voice_prompt_key = None
        self.ref_label.configure(text=os.path.basename(path))
        self.footer.configure(text="Preparing MDX-Net GPU separation...")
        self.download_reference_button.configure(state="disabled")
        self.play_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.check_button.configure(state="disabled")
        self.reload_button.configure(state="disabled")
        self.generate_button.configure(state="disabled")
        print(f"\nReference selected: {path}", flush=True)
        self.processing_reference = True
        self.ref_button.configure(state="disabled")
        self.progress.start()
        threading.Thread(target=self.process_reference, daemon=True).start()

    def load_model_thread(self):
        if self.loading or self.model is not None:
            return
        self.loading = True
        self.generate_button.configure(state="disabled")
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        self.loading = True
        self.after(0, self.progress.start)
        self.after(0, lambda: self.footer.configure(text="Loading Qwen3-TTS 1.7B 4-bit..."))
        print("\n========== QWEN3-TTS MODEL LOADING ==========", flush=True)
        print(f"Model: {MODEL_ID}", flush=True)
        print("Quantization: 4-bit NF4", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB", flush=True)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=[])
                self.model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", quantization_config=quant_config, dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True)
            else:
                self.model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True)
            print("========== MODEL LOADED SUCCESSFULLY ==========\n", flush=True)
            self.after(0, lambda: self.footer.configure(text="Qwen3-TTS ready"))
            return True
        except Exception:
            print("\n========== MODEL LOAD TRACEBACK ==========", flush=True)
            traceback.print_exc()
            print("========== END TRACEBACK ==========\n", flush=True)
            self.model = None
            self.after(0, lambda: self.footer.configure(text="Model loading failed - see terminal"))
            return False
        finally:
            self.loading = False
            self.after(0, self.progress.stop)

    def unload_qwen_for_mdx(self):
        if self.model is None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return
        print("Unloading Qwen3-TTS before MDX-Net GPU processing...", flush=True)
        self.voice_prompt = None
        self.voice_prompt_key = None
        qwen_model = self.model
        self.model = None
        try:
            del qwen_model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        print("Qwen3-TTS unloaded and CUDA memory cleared.", flush=True)

    def get_ffmpeg_path(self):
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()

    def convert_reference_to_wav(self, source_path):
        ffmpeg_path = self.get_ffmpeg_path()
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav = temp_file.name
        temp_file.close()
        try:
            command = [ffmpeg_path, "-y", "-i", source_path, "-vn", "-ac", "1", "-ar", str(REFERENCE_SAMPLE_RATE), "-sample_fmt", "s16", temp_wav]
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
            if result.returncode != 0:
                error_text = result.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg failed while extracting reference audio:\n{error_text}")
            audio, sr = sf.read(temp_wav, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            return np.ascontiguousarray(audio, dtype=np.float32), sr
        finally:
            try:
                os.unlink(temp_wav)
            except Exception:
                pass

    def load_enhancer(self):
        if self.enhancer is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MDX-Net in this configuration.")
        import onnxruntime as ort
        try:
            ort.preload_dlls()
        except Exception as preload_error:
            print(f"ONNX Runtime DLL preload warning: {preload_error}", flush=True)
        available_providers = ort.get_available_providers()
        print(f"ONNX Runtime version: {ort.__version__}", flush=True)
        print(f"ONNX Runtime providers: {available_providers}", flush=True)
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(f"CUDAExecutionProvider is not available. Installed ONNX Runtime providers: {available_providers}")
        from audio_separator.separator import Separator
        print(f"Loading MDX-Net model: {MDX_MODEL}", flush=True)
        print("MDX-Net device: CUDA", flush=True)
        print(f"MDX segment size: {MDX_SEGMENT_SIZE}", flush=True)
        print(f"MDX overlap: {MDX_OVERLAP}", flush=True)
        print(f"MDX batch size: {MDX_BATCH_SIZE}", flush=True)
        print(f"MDX denoise: {MDX_ENABLE_DENOISE}", flush=True)
        self.enhancer_device = "cuda"
        self.enhancer = Separator(model_file_dir=MDX_MODEL_DIR, output_dir=MDX_OUTPUT_DIR, output_format="WAV", output_single_stem="Vocals", sample_rate=SEPARATION_SAMPLE_RATE, use_soundfile=True, use_autocast=False, mdx_params={"hop_length": MDX_HOP_LENGTH, "segment_size": MDX_SEGMENT_SIZE, "overlap": MDX_OVERLAP, "batch_size": MDX_BATCH_SIZE, "enable_denoise": MDX_ENABLE_DENOISE})
        self.enhancer.load_model(model_filename=MDX_MODEL)
        effective_device = getattr(self.enhancer, "torch_device", None)
        effective_provider = getattr(self.enhancer, "onnx_execution_provider", None)
        print(f"MDX effective Torch device: {effective_device}", flush=True)
        print(f"MDX effective ONNX provider: {effective_provider}", flush=True)
        print("MDX-Net loaded successfully.", flush=True)

    def release_enhancer(self):
        if self.enhancer is not None:
            print("Releasing MDX-Net model...", flush=True)
            enhancer = self.enhancer
            self.enhancer = None
            self.enhancer_device = None
            try:
                del enhancer
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            except Exception:
                pass
        print("MDX-Net GPU memory released.", flush=True)

    def load_whisper(self):
        if self.whisper_model is not None:
            return
        from faster_whisper import WhisperModel
        print(f"Loading Whisper model: {WHISPER_MODEL}", flush=True)
        self.whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("Whisper model loaded.", flush=True)

    def resample_audio(self, audio, source_sr, target_sr):
        if source_sr == target_sr:
            return np.ascontiguousarray(audio, dtype=np.float32)
        gcd = np.gcd(source_sr, target_sr)
        up = target_sr // gcd
        down = source_sr // gcd
        resampled = resample_poly(audio, up, down)
        return np.ascontiguousarray(resampled, dtype=np.float32)

    def balance_reference_volume(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            raise RuntimeError("Reference audio is empty.")
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        peak = float(np.max(np.abs(audio)))
        if rms <= 1e-6 or peak <= 1e-6:
            raise RuntimeError("Reference audio is silent and cannot be normalized.")
        target_rms = 10.0 ** (REFERENCE_TARGET_RMS_DB / 20.0)
        peak_limit = 10.0 ** (REFERENCE_MAX_PEAK_DB / 20.0)
        gain = target_rms / rms
        if peak * gain > peak_limit:
            gain = peak_limit / peak
        audio = audio * gain
        final_peak = float(np.max(np.abs(audio)))
        if final_peak > peak_limit:
            audio = audio * (peak_limit / final_peak)
        audio = np.clip(audio, -1.0, 1.0)
        final_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        final_peak = float(np.max(np.abs(audio)))
        final_rms_db = 20.0 * np.log10(max(final_rms, 1e-8))
        final_peak_db = 20.0 * np.log10(max(final_peak, 1e-8))
        print(f"Normalized reference volume: RMS {final_rms_db:.2f} dBFS, peak {final_peak_db:.2f} dBFS.", flush=True)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def stabilize_mdx_output(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 0.0:
            raise RuntimeError("MDX-Net returned an empty or silent vocal waveform.")
        peak_limit = 10.0 ** (REFERENCE_MAX_PEAK_DB / 20.0)
        if peak > peak_limit:
            audio = audio * (peak_limit / peak)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def separate_speech(self, waveform_24k):
        waveform_24k = np.asarray(waveform_24k, dtype=np.float32)
        input_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_input = input_file.name
        input_file.close()
        expected_path = os.path.join(MDX_OUTPUT_DIR, "mdx_vocals.wav")
        try:
            if os.path.isfile(expected_path):
                try:
                    os.remove(expected_path)
                except Exception:
                    pass
            sf.write(temp_input, waveform_24k, REFERENCE_SAMPLE_RATE, subtype="PCM_16")
            print("Running UVR-MDX-NET-Voc_FT on CUDA.", flush=True)
            output_files = self.enhancer.separate(temp_input, {"Vocals": "mdx_vocals"})
            if not output_files:
                raise RuntimeError("MDX-Net returned no separated files.")
            print(f"MDX-Net returned output files: {output_files}", flush=True)
            vocal_path = None
            for candidate in output_files:
                if not candidate:
                    continue
                candidate_path = candidate if os.path.isabs(candidate) else os.path.join(MDX_OUTPUT_DIR, candidate)
                if os.path.isfile(candidate_path):
                    if "vocal" in os.path.basename(candidate_path).lower():
                        vocal_path = candidate_path
                        break
                    if vocal_path is None:
                        vocal_path = candidate_path
            if vocal_path is None and os.path.isfile(expected_path):
                vocal_path = expected_path
            if vocal_path is None:
                wav_candidates = [os.path.join(MDX_OUTPUT_DIR, name) for name in os.listdir(MDX_OUTPUT_DIR) if name.lower().endswith(".wav")]
                if wav_candidates:
                    wav_candidates.sort(key=os.path.getmtime, reverse=True)
                    vocal_path = wav_candidates[0]
            if vocal_path is None:
                raise RuntimeError(f"MDX-Net completed but no output WAV could be located in {MDX_OUTPUT_DIR}. Returned paths: {output_files}")
            print(f"Reading MDX-Net vocal output: {vocal_path}", flush=True)
            audio, sr = sf.read(vocal_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            if sr != SEPARATION_SAMPLE_RATE:
                audio = self.resample_audio(audio, sr, SEPARATION_SAMPLE_RATE)
                sr = SEPARATION_SAMPLE_RATE
            audio = self.stabilize_mdx_output(audio)
            rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
            peak = float(np.max(np.abs(audio)))
            rms_db = 20.0 * np.log10(max(rms, 1e-8))
            peak_db = 20.0 * np.log10(max(peak, 1e-8))
            print(f"MDX-Net output sample rate: {sr} Hz", flush=True)
            print(f"MDX-Net output duration: {len(audio) / SEPARATION_SAMPLE_RATE:.2f} seconds", flush=True)
            print(f"MDX-Net output RMS: {rms_db:.2f} dBFS", flush=True)
            print(f"MDX-Net output peak: {peak_db:.2f} dBFS", flush=True)
            return audio
        finally:
            try:
                os.unlink(temp_input)
            except Exception:
                pass

    def detect_speech_regions(self, audio, sr):
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            return []
        peak = float(np.max(np.abs(audio)))
        rms_total = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        if peak <= 1e-7 or rms_total <= 1e-8:
            return []
        frame_size = max(1, int(sr * SILENCE_FRAME_MS / 1000.0))
        hop_size = max(1, int(sr * SILENCE_HOP_MS / 1000.0))
        if len(audio) < frame_size:
            rms = np.array([np.sqrt(np.mean(audio * audio) + 1e-10)], dtype=np.float32)
        else:
            frame_count = 1 + (len(audio) - frame_size) // hop_size
            rms = np.empty(frame_count, dtype=np.float32)
            for index in range(frame_count):
                start = index * hop_size
                frame = audio[start:start + frame_size]
                rms[index] = np.sqrt(np.mean(frame * frame) + 1e-10)
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
        peak_db = 20.0 * np.log10(max(peak, 1e-8))
        noise_floor_db = float(np.percentile(rms_db, 20))
        adaptive_threshold_db = noise_floor_db + SILENCE_NOISE_MARGIN_DB
        relative_threshold_db = peak_db - SILENCE_RELATIVE_THRESHOLD_DB
        threshold_db = max(SILENCE_THRESHOLD_DB, min(adaptive_threshold_db, relative_threshold_db))
        print(f"Speech detector total RMS: {20.0 * np.log10(max(rms_total, 1e-8)):.2f} dBFS", flush=True)
        print(f"Speech detector peak: {peak_db:.2f} dBFS", flush=True)
        print(f"Speech detector noise floor: {noise_floor_db:.2f} dBFS", flush=True)
        print(f"Speech detector threshold: {threshold_db:.2f} dBFS", flush=True)
        active = rms_db > threshold_db
        regions = []
        region_start = None
        for index, is_active in enumerate(active):
            if is_active and region_start is None:
                region_start = index
            elif not is_active and region_start is not None:
                start_sample = region_start * hop_size
                end_sample = min(len(audio), index * hop_size + frame_size)
                if end_sample > start_sample:
                    regions.append((start_sample, end_sample))
                region_start = None
        if region_start is not None:
            start_sample = region_start * hop_size
            end_sample = len(audio)
            if end_sample > start_sample:
                regions.append((start_sample, end_sample))
        minimum_samples = int(sr * MIN_SPEECH_SECONDS)
        filtered = [(start, end) for start, end in regions if end - start >= minimum_samples]
        if filtered:
            return filtered
        fallback_threshold_db = max(SILENCE_THRESHOLD_DB, peak_db - 36.0)
        print(f"Speech detector primary pass found no regions. Fallback threshold: {fallback_threshold_db:.2f} dBFS", flush=True)
        fallback_active = rms_db > fallback_threshold_db
        fallback_regions = []
        region_start = None
        for index, is_active in enumerate(fallback_active):
            if is_active and region_start is None:
                region_start = index
            elif not is_active and region_start is not None:
                start_sample = region_start * hop_size
                end_sample = min(len(audio), index * hop_size + frame_size)
                if end_sample > start_sample:
                    fallback_regions.append((start_sample, end_sample))
                region_start = None
        if region_start is not None:
            start_sample = region_start * hop_size
            end_sample = len(audio)
            if end_sample > start_sample:
                fallback_regions.append((start_sample, end_sample))
        filtered_fallback = [(start, end) for start, end in fallback_regions if end - start >= minimum_samples]
        if filtered_fallback:
            print(f"Speech detector fallback found {len(filtered_fallback)} region(s).", flush=True)
            return filtered_fallback
        minimum_reference_rms_db = -65.0
        total_rms_db = 20.0 * np.log10(max(rms_total, 1e-8))
        if total_rms_db > minimum_reference_rms_db:
            print(f"Speech detector fallback accepted the complete MDX vocal because total RMS {total_rms_db:.2f} dBFS is above {minimum_reference_rms_db:.2f} dBFS.", flush=True)
            return [(0, len(audio))]
        return []

    def merge_speech_regions(self, regions, sr):
        if not regions:
            return []
        split_gap_samples = int(sr * SILENCE_SPLIT_GAP_SECONDS)
        merged = [regions[0]]
        for start, end in regions[1:]:
            previous_start, previous_end = merged[-1]
            gap_samples = start - previous_end
            if gap_samples < split_gap_samples:
                merged[-1] = (previous_start, end)
            else:
                merged.append((start, end))
        return merged

    def trim_long_internal_silences(self, audio, sr, regions):
        return np.ascontiguousarray(audio, dtype=np.float32)

    def trim_reference_edges_only(self, audio, sr):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("Reference audio is empty.")
        detected_regions = self.detect_speech_regions(audio, sr)
        if not detected_regions:
            raise RuntimeError("No usable vocal energy was detected in the MDX-Net reference. Try a clearer voice sample.")
        regions = self.merge_speech_regions(detected_regions, sr)
        first_start = max(0, regions[0][0] - int(sr * LEADING_TRAILING_PADDING_SECONDS))
        last_end = min(len(audio), regions[-1][1] + int(sr * LEADING_TRAILING_PADDING_SECONDS))
        base_audio = np.ascontiguousarray(audio[first_start:last_end], dtype=np.float32)
        original_duration = len(audio) / sr
        final_duration = len(base_audio) / sr
        removed_duration = original_duration - final_duration
        print(f"Detected speech regions: {len(detected_regions)}", flush=True)
        print(f"Detected grouped regions: {len(regions)}", flush=True)
        print(f"Internal silence trimming: DISABLED", flush=True)
        print(f"All internal pauses are preserved exactly.", flush=True)
        print(f"Reference duration before edge trim: {original_duration:.3f}s", flush=True)
        print(f"Reference duration after edge trim: {final_duration:.3f}s", flush=True)
        print(f"Leading/trailing duration removed: {max(0.0, removed_duration):.3f}s", flush=True)
        if len(regions) > 1:
            gaps = []
            for index in range(len(regions) - 1):
                gap_seconds = (regions[index + 1][0] - regions[index][1]) / sr
                gaps.append(gap_seconds)
            print(f"Internal pauses preserved: {[round(value, 3) for value in gaps]}", flush=True)
        return base_audio

    def transcribe_audio(self, audio, sr):
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        try:
            sf.write(temp_path, audio, sr, subtype="PCM_16")
            segments, info = self.whisper_model.transcribe(temp_path, beam_size=5, best_of=5, temperature=0.0, vad_filter=False, condition_on_previous_text=False, compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6)
            parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def split_text_for_generation(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        if len(text) <= MAX_SINGLE_GENERATION_CHARS:
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        segments = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not current:
                current = sentence
                continue
            if len(current) + 1 + len(sentence) <= SEGMENT_MAX_CHARS:
                current = f"{current} {sentence}"
                continue
            segments.append(current)
            current = sentence
        if current:
            segments.append(current)
        final_segments = []
        for segment in segments:
            if len(segment) <= SEGMENT_MAX_CHARS:
                final_segments.append(segment)
                continue
            words = segment.split()
            chunk = ""
            for word in words:
                if not chunk:
                    chunk = word
                elif len(chunk) + 1 + len(word) <= SEGMENT_MAX_CHARS:
                    chunk = f"{chunk} {word}"
                else:
                    if chunk:
                        final_segments.append(chunk)
                    chunk = word
            if chunk:
                final_segments.append(chunk)
        merged_segments = []
        for segment in final_segments:
            if merged_segments and len(merged_segments[-1]) < SEGMENT_MIN_CHARS and len(merged_segments[-1]) + 1 + len(segment) <= SEGMENT_MAX_CHARS:
                merged_segments[-1] = f"{merged_segments[-1]} {segment}"
            else:
                merged_segments.append(segment)
        return merged_segments

    def calculate_max_new_tokens(self, text):
        text_length = len(text.strip())
        estimated_tokens = int(text_length * 14)
        return max(512, min(estimated_tokens, GENERATION_SINGLE_MAX_NEW_TOKENS))

    def calculate_segment_max_new_tokens(self, text):
        text_length = len(text.strip())
        estimated_tokens = int(text_length * 18)
        return max(256, min(estimated_tokens, GENERATION_SEGMENT_MAX_NEW_TOKENS))

    def reload_reference_transcript(self):
        if self.clean_reference_audio is None or self.clean_reference_sr is None:
            self.footer.configure(text="Process a reference sample first")
            return
        if self.transcribing_reference or self.processing_reference or self.generating or self.loading:
            return
        threading.Thread(target=self.reload_reference_transcript_worker, daemon=True).start()

    def reload_reference_transcript_worker(self):
        self.transcribing_reference = True
        self.after(0, lambda: self.reload_button.configure(state="disabled"))
        self.after(0, lambda: self.check_button.configure(state="disabled"))
        self.after(0, lambda: self.ref_button.configure(state="disabled"))
        self.after(0, lambda: self.generate_button.configure(state="disabled"))
        self.after(0, self.progress.start)
        self.after(0, lambda: self.footer.configure(text="Reloading reference transcript..."))
        try:
            self.load_whisper()
            transcript = self.transcribe_audio(self.clean_reference_audio, self.clean_reference_sr)
            if not transcript:
                raise RuntimeError("Whisper returned an empty transcript.")
            print(f"\nReloaded reference transcript: {transcript}", flush=True)
            self.voice_prompt = None
            self.voice_prompt_key = None
            self.after(0, self.set_reference_transcript, transcript)
            self.after(0, lambda: self.footer.configure(text="Reference transcript reloaded - verify every word"))
        except Exception:
            print("\n========== TRANSCRIPT RELOAD TRACEBACK ==========", flush=True)
            traceback.print_exc()
            print("========== END TRANSCRIPT RELOAD TRACEBACK ==========\n", flush=True)
            self.after(0, lambda: self.footer.configure(text="Transcript reload failed - see terminal"))
        finally:
            self.transcribing_reference = False
            self.after(0, self.progress.stop)
            self.after(0, lambda: self.reload_button.configure(state="normal" if self.clean_reference_audio is not None else "disabled"))
            self.after(0, lambda: self.check_button.configure(state="normal" if self.clean_reference_audio is not None else "disabled"))
            self.after(0, lambda: self.download_reference_button.configure(state="normal" if self.clean_reference_audio is not None else "disabled"))
            self.after(0, lambda: self.ref_button.configure(state="normal"))
            self.after(0, lambda: self.generate_button.configure(state="normal" if self.model is not None and self.clean_reference_audio is not None else "disabled"))

    def process_reference(self):
        started = time.time()
        qwen_was_loaded = self.model is not None
        try:
            print("\n========== REFERENCE PROCESSING ==========", flush=True)
            self.after(0, lambda: self.footer.configure(text="Extracting reference at 24 kHz..."))
            original_24k, sr = self.convert_reference_to_wav(self.reference_path)
            original_duration = len(original_24k) / sr
            print(f"Extracted sample rate: {sr} Hz", flush=True)
            print(f"Extracted duration: {original_duration:.2f} seconds", flush=True)
            if original_duration < REFERENCE_MIN_SECONDS:
                raise RuntimeError(f"Reference audio is too short. Use at least {REFERENCE_MIN_SECONDS:.1f} seconds.")
            if original_duration > REFERENCE_MAX_SECONDS:
                print(f"Reference is longer than {REFERENCE_MAX_SECONDS:.1f} seconds and will not be hard-truncated.", flush=True)
            if qwen_was_loaded:
                self.after(0, lambda: self.footer.configure(text="Releasing Qwen GPU memory for MDX-Net..."))
                self.unload_qwen_for_mdx()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self.after(0, lambda: self.footer.configure(text="Running UVR-MDX-NET-Voc_FT on RTX 2050..."))
            self.load_enhancer()
            separated_44k = self.separate_speech(original_24k)
            mdx_duration = len(separated_44k) / SEPARATION_SAMPLE_RATE
            print(f"MDX-Net vocal duration: {mdx_duration:.2f} seconds", flush=True)
            self.after(0, lambda: self.footer.configure(text="Releasing MDX-Net GPU memory..."))
            self.release_enhancer()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self.after(0, lambda: self.footer.configure(text="Trimming only leading and trailing silence..."))
            trimmed_44k = self.trim_reference_edges_only(separated_44k, SEPARATION_SAMPLE_RATE)
            balanced_44k = self.balance_reference_volume(trimmed_44k)
            final_reference = self.resample_audio(balanced_44k, SEPARATION_SAMPLE_RATE, REFERENCE_SAMPLE_RATE)
            final_reference = self.balance_reference_volume(final_reference)
            self.clean_reference_audio = final_reference
            self.clean_reference_sr = REFERENCE_SAMPLE_RATE
            self.voice_prompt = None
            self.voice_prompt_key = None
            final_rms = float(np.sqrt(np.mean(self.clean_reference_audio * self.clean_reference_audio) + 1e-12))
            final_peak = float(np.max(np.abs(self.clean_reference_audio)))
            final_rms_db = 20.0 * np.log10(max(final_rms, 1e-8))
            final_peak_db = 20.0 * np.log10(max(final_peak, 1e-8))
            final_duration = len(self.clean_reference_audio) / self.clean_reference_sr
            print(f"Final Qwen reference RMS: {final_rms_db:.2f} dBFS", flush=True)
            print(f"Final Qwen reference peak: {final_peak_db:.2f} dBFS", flush=True)
            print(f"Final Qwen reference sample rate: {self.clean_reference_sr} Hz", flush=True)
            print(f"Final Qwen reference duration: {final_duration:.2f} seconds", flush=True)
            self.after(0, lambda: self.download_reference_button.configure(state="normal"))
            self.after(0, lambda: self.check_button.configure(state="normal"))
            if qwen_was_loaded:
                self.after(0, lambda: self.footer.configure(text="Reloading Qwen3-TTS..."))
                if not self.load_model():
                    raise RuntimeError("Qwen3-TTS could not be reloaded after MDX-Net processing.")
            self.after(0, lambda: self.footer.configure(text="Transcribing final reference..."))
            self.load_whisper()
            transcript = self.transcribe_audio(self.clean_reference_audio, self.clean_reference_sr)
            if not transcript:
                raise RuntimeError("Whisper returned an empty transcript for the cleaned MDX-Net reference.")
            print(f"Auto transcript: {transcript}", flush=True)
            self.after(0, self.set_reference_transcript, transcript)
            elapsed = time.time() - started
            print(f"Reference processing time: {elapsed:.2f} seconds", flush=True)
            print("========== REFERENCE PROCESSING SUCCESS ==========\n", flush=True)
            self.after(0, lambda: self.footer.configure(text=f"Reference ready in {elapsed:.1f}s - verify transcript"))
        except Exception:
            print("\n========== REFERENCE PROCESSING TRACEBACK ==========", flush=True)
            traceback.print_exc()
            print("========== END REFERENCE PROCESSING TRACEBACK ==========\n", flush=True)
            self.release_enhancer()
            self.clean_reference_audio = None
            self.clean_reference_sr = None
            self.voice_prompt = None
            self.voice_prompt_key = None
            if qwen_was_loaded and self.model is None:
                try:
                    self.load_model()
                except Exception:
                    traceback.print_exc()
            self.after(0, lambda: self.download_reference_button.configure(state="disabled"))
            self.after(0, lambda: self.check_button.configure(state="disabled"))
            self.after(0, lambda: self.reload_button.configure(state="disabled"))
            self.after(0, lambda: self.footer.configure(text="Reference processing failed - see terminal"))
        finally:
            self.processing_reference = False
            self.release_enhancer()
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            self.after(0, self.progress.stop)
            self.after(0, lambda: self.ref_button.configure(state="normal"))
            self.after(0, lambda: self.check_button.configure(state="normal" if self.clean_reference_audio is not None else "disabled"))
            self.after(0, lambda: self.reload_button.configure(state="normal" if self.clean_reference_audio is not None else "disabled"))
            self.after(0, lambda: self.download_reference_button.configure(state="normal" if self.clean_reference_audio is not None else "disabled"))
            self.after(0, lambda: self.generate_button.configure(state="normal" if self.model is not None and self.clean_reference_audio is not None else "disabled"))

    def set_reference_transcript(self, transcript):
        self.ref_text_box.delete("1.0", "end")
        self.ref_text_box.insert("1.0", transcript)

    def check_voice_sample(self):
        if self.clean_reference_audio is None or self.clean_reference_sr is None:
            print("Check requested but no processed reference exists.", flush=True)
            return
        try:
            sd.stop()
            sd.play(self.clean_reference_audio, self.clean_reference_sr)
            print(f"Processed reference playback started at {self.clean_reference_sr} Hz.", flush=True)
            self.footer.configure(text="Playing exact normalized MDX-Net reference")
        except Exception:
            print("\n========== REFERENCE PLAYBACK TRACEBACK ==========", flush=True)
            traceback.print_exc()
            print("========== END PLAYBACK TRACEBACK ==========\n", flush=True)

    def download_voice_sample(self):
        if self.clean_reference_audio is None or self.clean_reference_sr is None:
            print("Voice sample download requested but no cleaned reference exists.", flush=True)
            self.footer.configure(text="No cleaned voice sample available")
            return
        path = filedialog.asksaveasfilename(title="Download cleaned voice sample", defaultextension=".wav", initialfile="cleaned_voice_sample.wav", filetypes=[("WAV Audio", "*.wav")])
        if not path:
            return
        try:
            sf.write(path, self.clean_reference_audio, self.clean_reference_sr, subtype="PCM_16")
            self.footer.configure(text=f"Saved cleaned voice sample: {path}")
            print(f"Cleaned voice sample saved: {path}", flush=True)
        except Exception:
            print("\n========== CLEANED VOICE SAMPLE SAVE TRACEBACK ==========", flush=True)
            traceback.print_exc()
            print("========== END CLEANED VOICE SAMPLE SAVE TRACEBACK ==========\n", flush=True)
            self.footer.configure(text="Cleaned voice sample save failed - see terminal")

    def generate_thread(self):
        if self.generating:
            return
        if self.loading:
            print("Generation blocked because the model is still loading.", flush=True)
            return
        if self.processing_reference or self.transcribing_reference:
            print("Generation blocked because the reference is still being processed.", flush=True)
            return
        if self.model is None:
            print("Generation blocked because the model is not loaded.", flush=True)
            self.after(0, lambda: self.footer.configure(text="Model not loaded"))
            return
        if self.clean_reference_audio is None:
            print("Generation blocked because no reference exists.", flush=True)
            self.after(0, lambda: self.footer.configure(text="Select and process a voice sample first"))
            return
        threading.Thread(target=self.generate, daemon=True).start()

    def update_generation_status(self):
        if not self.generating:
            return
        elapsed = time.time() - self.generate_start_time
        self.after(0, lambda: self.footer.configure(text=f"Generating speech... {elapsed:.0f}s elapsed"))
        self.after(1000, self.update_generation_status)

    def create_voice_prompt(self, ref_text):
        prompt_key = (self.reference_path, ref_text, self.clean_reference_sr, len(self.clean_reference_audio), float(np.mean(np.abs(self.clean_reference_audio))), float(np.max(np.abs(self.clean_reference_audio))))
        if self.voice_prompt is None or self.voice_prompt_key != prompt_key:
            print("Creating reusable ICL voice clone prompt list...", flush=True)
            prompt_items = self.model.create_voice_clone_prompt(ref_audio=(self.clean_reference_audio, self.clean_reference_sr), ref_text=ref_text, x_vector_only_mode=False)
            if not prompt_items:
                raise RuntimeError("Qwen did not return a voice clone prompt.")
            self.voice_prompt = prompt_items
            self.voice_prompt_key = prompt_key
            print(f"Reusable prompt count: {len(self.voice_prompt)}", flush=True)
            print(f"Reusable prompt item type: {type(self.voice_prompt[0]).__name__}", flush=True)
            print("Reusable ICL voice clone prompt created.", flush=True)

    def generate_single_text(self, text, language):
        max_new_tokens = self.calculate_max_new_tokens(text)
        print("Generating the entire passage as one Qwen sequence.", flush=True)
        print(f"Text length: {len(text)} characters", flush=True)
        print(f"max_new_tokens: {max_new_tokens}", flush=True)
        with torch.inference_mode():
            wavs, sr = self.model.generate_voice_clone(text=[text], language=[language], voice_clone_prompt=self.voice_prompt, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=GENERATION_REPETITION_PENALTY, non_streaming_mode=True)
        if not wavs:
            raise RuntimeError("Qwen returned no audio.")
        audio = np.asarray(wavs[0], dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            raise RuntimeError("Qwen returned empty audio.")
        return audio, sr

    def generate_batch(self, segments, language):
        output_parts = []
        output_sr = None
        total_segments = len(segments)
        batch_size = min(MAX_BATCH_SIZE, total_segments)
        total_batches = (total_segments + batch_size - 1) // batch_size
        for batch_start in range(0, total_segments, batch_size):
            batch_segments = segments[batch_start:batch_start + batch_size]
            batch_languages = [language] * len(batch_segments)
            batch_limits = [self.calculate_segment_max_new_tokens(segment) for segment in batch_segments]
            batch_max_tokens = max(batch_limits)
            batch_number = (batch_start // batch_size) + 1
            print(f"Generating batch {batch_number}/{total_batches} with {len(batch_segments)} segment(s).", flush=True)
            for local_index, segment in enumerate(batch_segments):
                print(f"Segment {batch_start + local_index + 1}/{total_segments}: {segment}", flush=True)
            print(f"Batch max_new_tokens: {batch_max_tokens}", flush=True)
            with torch.inference_mode():
                wavs, sr = self.model.generate_voice_clone(text=batch_segments, language=batch_languages, voice_clone_prompt=self.voice_prompt, max_new_tokens=batch_max_tokens, do_sample=False, repetition_penalty=GENERATION_REPETITION_PENALTY, non_streaming_mode=True)
            if output_sr is None:
                output_sr = sr
            for index, audio in enumerate(wavs):
                audio = np.asarray(audio, dtype=np.float32)
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                if audio.size == 0:
                    raise RuntimeError(f"Qwen returned empty audio for segment {batch_start + index + 1}.")
                if sr != output_sr:
                    audio = self.resample_audio(audio, sr, output_sr)
                output_parts.append(audio)
                if batch_start + index < total_segments - 1:
                    output_parts.append(np.zeros(int(output_sr * SEGMENT_JOIN_PAUSE_SECONDS), dtype=np.float32))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not output_parts:
            raise RuntimeError("No audio was generated.")
        return np.concatenate(output_parts), output_sr

    def generate(self):
        if self.model is None:
            print("Generation requested before model was loaded.", flush=True)
            return
        if self.clean_reference_audio is None:
            print("Generation requested without reference audio.", flush=True)
            return
        text = self.text_box.get("1.0", "end").strip()
        ref_text = self.ref_text_box.get("1.0", "end").strip()
        language = self.language_menu.get()
        if not text:
            print("Generation requested with empty text.", flush=True)
            self.after(0, lambda: self.footer.configure(text="Text required"))
            return
        if not ref_text:
            print("Generation requested without reference transcript.", flush=True)
            self.after(0, lambda: self.footer.configure(text="Reference transcript required"))
            return
        segments = self.split_text_for_generation(text)
        if not segments:
            print("Text splitting produced no segments.", flush=True)
            self.after(0, lambda: self.footer.configure(text="No valid generation text"))
            return
        self.generating = True
        self.generate_start_time = time.time()
        self.after(0, lambda: self.generate_button.configure(state="disabled"))
        self.after(0, lambda: self.ref_button.configure(state="disabled"))
        self.after(0, lambda: self.check_button.configure(state="disabled"))
        self.after(0, lambda: self.reload_button.configure(state="disabled"))
        self.after(0, lambda: self.download_reference_button.configure(state="disabled"))
        self.after(0, self.progress.start)
        self.after(0, self.update_generation_status)
        print("\n========== QWEN3-TTS MAXIMUM STABILITY GENERATION ==========", flush=True)
        print(f"Model: {MODEL_ID}", flush=True)
        print(f"Language: {language}", flush=True)
        print(f"Reference: {self.reference_path}", flush=True)
        print(f"Reference sample rate: {self.clean_reference_sr} Hz", flush=True)
        print(f"Reference duration: {len(self.clean_reference_audio) / self.clean_reference_sr:.2f} seconds", flush=True)
        print("ICL mode: True", flush=True)
        print("X-vector-only mode: False", flush=True)
        print(f"Input text length: {len(text)} characters", flush=True)
        print(f"Generation units: {len(segments)}", flush=True)
        print(f"Single-generation threshold: {MAX_SINGLE_GENERATION_CHARS} characters", flush=True)
        print("Sampling disabled for maximum speaker consistency.", flush=True)
        print("Non-streaming mode enabled.", flush=True)
        print("Qwen precision: 4-bit NF4", flush=True)
        print("Qwen device: CUDA", flush=True)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.create_voice_prompt(ref_text)
            if len(segments) == 1:
                generated_audio, generated_sr = self.generate_single_text(segments[0], language)
            else:
                generated_audio, generated_sr = self.generate_batch(segments, language)
            self.generated_audio = np.ascontiguousarray(generated_audio, dtype=np.float32)
            self.generated_sr = generated_sr
            duration = len(self.generated_audio) / self.generated_sr
            elapsed = time.time() - self.generate_start_time
            print(f"Generated audio duration: {duration:.2f} seconds", flush=True)
            print(f"Generation sample rate: {self.generated_sr} Hz", flush=True)
            print(f"Generation time: {elapsed:.2f} seconds", flush=True)
            print("========== GENERATION SUCCESS ==========\n", flush=True)
            self.after(0, lambda: self.footer.configure(text=f"Generated {duration:.1f}s in {elapsed:.1f}s"))
            self.after(0, lambda: self.play_button.configure(state="normal"))
            self.after(0, lambda: self.save_button.configure(state="normal"))
        except Exception:
            print("\n========== QWEN3-TTS GENERATION TRACEBACK ==========", flush=True)
            traceback.print_exc()
            print("========== END QWEN3-TTS GENERATION TRACEBACK ==========\n", flush=True)
            self.after(0, lambda: self.footer.configure(text="Generation failed - see terminal"))
        finally:
            self.generating = False
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            self.after(0, self.progress.stop)
            self.after(0, lambda: self.generate_button.configure(state="normal" if self.model is not None and self.clean_reference_audio is not None else "disabled"))
            self.after(0, lambda: self.ref_button.configure(state="normal"))
            self.after(0, lambda: self.check_button.configure(state="normal" if self.clean_reference_audio is not None else "disabled"))
            self.after(0, lambda: self.reload_button.configure(state="normal" if self.clean_reference_audio is not None else "disabled"))
            self.after(0, lambda: self.download_reference_button.configure(state="normal" if self.clean_reference_audio is not None else "disabled"))

    def play_audio(self):
        if self.generated_audio is None or self.generated_sr is None:
            print("Playback requested but no generated audio exists.", flush=True)
            return
        try:
            sd.stop()
            sd.play(self.generated_audio, self.generated_sr)
            print("Generated audio playback started.", flush=True)
        except Exception:
            print("\n========== PLAYBACK TRACEBACK ==========", flush=True)
            traceback.print_exc()
            print("========== END PLAYBACK TRACEBACK ==========\n", flush=True)

    def save_audio(self):
        if self.generated_audio is None or self.generated_sr is None:
            print("Save requested but no generated audio exists.", flush=True)
            return
        path = filedialog.asksaveasfilename(title="Save generated speech", defaultextension=".wav", filetypes=[("WAV Audio", "*.wav")])
        if not path:
            return
        try:
            sf.write(path, self.generated_audio, self.generated_sr, subtype="PCM_16")
            self.footer.configure(text=f"Saved: {path}")
            print(f"Audio saved: {path}", flush=True)
        except Exception:
            print("\n========== SAVE TRACEBACK ==========", flush=True)
            traceback.print_exc()
            print("========== END SAVE TRACEBACK ==========\n", flush=True)

    def clear_all(self):
        sd.stop()
        self.text_box.delete("1.0", "end")
        self.ref_text_box.delete("1.0", "end")
        self.reference_path = None
        self.clean_reference_audio = None
        self.clean_reference_sr = None
        self.voice_prompt = None
        self.voice_prompt_key = None
        self.generated_audio = None
        self.generated_sr = None
        self.ref_label.configure(text="No reference audio selected")
        self.footer.configure(text="Ready")
        self.check_button.configure(state="disabled")
        self.reload_button.configure(state="disabled")
        self.download_reference_button.configure(state="disabled")
        self.play_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.generate_button.configure(state="normal" if self.model is not None else "disabled")

    def on_close(self):
        try:
            sd.stop()
        except Exception:
            traceback.print_exc()
        self.release_enhancer()
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.destroy()

if __name__ == "__main__":
    app = QwenTTSApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()