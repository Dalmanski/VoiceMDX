import customtkinter as ctk
import asyncio
import edge_tts
import subprocess
import threading
import platform
import shutil
import tempfile
import time
import pygame
from pathlib import Path
from tkinter import filedialog

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LANGUAGE = "en-US"
DEFAULT_ENGINE = "Microsoft Edge Neural"
PLACEHOLDER_TEXT = "Type something here..."


class GridSelector:
    def __init__(self, master, values=None, command=None, width=400, height=38, columns=4, gender_grouped=False):
        self.master = master
        self.values = values or []
        self.items = []
        self.command = command
        self.width = width
        self.height = height
        self.columns = columns
        self.gender_grouped = gender_grouped
        self.value = ""
        self.popup = None
        self.search_entry = None
        self.scroll_frame = None
        self.button = ctk.CTkButton(master, text="", width=width, height=height, anchor="w", command=self.open)

    def grid(self, **kwargs):
        self.button.grid(**kwargs)

    def _label_for(self, value):
        for item in self.items:
            if item["value"] == value:
                return item["display"]
        return value

    def set(self, value):
        self.value = value
        self.button.configure(text=self._label_for(value))

    def get(self):
        return self.value

    def configure(self, **kwargs):
        if "values" in kwargs:
            self.values = kwargs["values"] or []
            self.items = [{"value": v, "display": v, "gender": "Unknown"} for v in self.values]
        if "items" in kwargs:
            self.items = kwargs["items"] or []
            self.values = [i["value"] for i in self.items]
        button_kwargs = {k: v for k, v in kwargs.items() if k not in ("values", "items")}
        if button_kwargs:
            self.button.configure(**button_kwargs)

    def open(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = ctk.CTkToplevel(self.master)
        self.popup.title("Select")
        self.popup.transient(self.master.winfo_toplevel())
        self.popup.grab_set()
        self.popup.resizable(False, True)
        popup_width = 1200
        item_count = max(1, len(self.items))
        popup_height = min(max(300, 220 + min(item_count, 16) * 28), 620)
        self.center_popup(popup_width, popup_height)
        self.popup.grid_rowconfigure(0, weight=0)
        self.popup.grid_rowconfigure(1, weight=0)
        self.popup.grid_rowconfigure(2, weight=1)
        self.popup.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(self.popup, text="Select", font=ctk.CTkFont(size=17, weight="bold"))
        title.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        search_frame = ctk.CTkFrame(self.popup, fg_color="transparent", border_width=0)
        search_frame.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="ew")
        search_frame.grid_columnconfigure(0, weight=1)
        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search...")
        self.search_entry.grid(row=0, column=0, sticky="ew")
        self.search_entry.bind("<KeyRelease>", self.filter_items)
        self.scroll_frame = ctk.CTkScrollableFrame(self.popup, width=popup_width - 30, height=popup_height - 115, corner_radius=12)
        self.scroll_frame.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.configure_columns()
        self.render_items()
        self.popup.after_idle(self.refresh_scroll_region)
        self.popup.after(50, self.refresh_scroll_region)
        self.popup.after(150, self.refresh_scroll_region)

    def center_popup(self, width, height):
        self.popup.update_idletasks()
        screen_width = self.popup.winfo_screenwidth()
        screen_height = self.popup.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.popup.geometry(f"{width}x{height}+{x}+{y}")

    def refresh_scroll_region(self):
        if self.scroll_frame is None or not self.scroll_frame.winfo_exists():
            return
        self.scroll_frame.update_idletasks()

    def filter_items(self, event=None):
        self.render_items()

    def get_filtered_items(self):
        search_text = self.search_entry.get().strip().lower() if self.search_entry is not None else ""
        if not search_text:
            return self.items
        return [item for item in self.items if search_text in item["value"].lower() or search_text in item["display"].lower() or search_text in item["gender"].lower()]

    def clear_grid(self):
        if self.scroll_frame is None:
            return
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

    def configure_columns(self):
        for column in range(self.columns):
            self.scroll_frame.grid_columnconfigure(column, weight=1, uniform="select_column")

    def render_items(self):
        if self.scroll_frame is None or not self.scroll_frame.winfo_exists():
            return
        self.clear_grid()
        self.configure_columns()
        filtered_items = self.get_filtered_items()
        if self.gender_grouped:
            self.render_gender_groups(filtered_items)
        else:
            self.render_standard_grid(filtered_items)
        self.scroll_frame.update_idletasks()

    def create_button(self, item):
        return ctk.CTkButton(self.scroll_frame, text=item["display"], height=40, command=lambda value=item["value"]: self.select(value))

    def render_standard_grid(self, items):
        if not items:
            empty_label = ctk.CTkLabel(self.scroll_frame, text="No matches found.", text_color="gray")
            empty_label.grid(row=0, column=0, columnspan=self.columns, padx=10, pady=20)
            return
        for index, item in enumerate(items):
            row = index // self.columns
            column = index % self.columns
            button = self.create_button(item)
            button.grid(row=row, column=column, padx=5, pady=5, sticky="ew")

    def render_gender_groups(self, items):
        groups = [("Female", [item for item in items if item["gender"] == "Female"]), ("Male", [item for item in items if item["gender"] == "Male"]), ("Other", [item for item in items if item["gender"] not in ("Female", "Male", "Unknown")]), ("Unknown", [item for item in items if item["gender"] == "Unknown"])]
        current_row = 0
        found_group = False
        for group_name, group_items in groups:
            if not group_items:
                continue
            found_group = True
            group_label = ctk.CTkLabel(self.scroll_frame, text=f"{group_name} Voices", font=ctk.CTkFont(size=14, weight="bold"))
            group_label.grid(row=current_row, column=0, columnspan=self.columns, padx=5, pady=(10, 5), sticky="w")
            current_row += 1
            for index, item in enumerate(group_items):
                row = current_row + index // self.columns
                column = index % self.columns
                button = self.create_button(item)
                button.grid(row=row, column=column, padx=5, pady=5, sticky="ew")
            current_row += (len(group_items) + self.columns - 1) // self.columns
        if not found_group:
            empty_label = ctk.CTkLabel(self.scroll_frame, text="No matches found.", text_color="gray")
            empty_label.grid(row=0, column=0, columnspan=self.columns, padx=10, pady=20)

    def select(self, value):
        self.value = value
        self.button.configure(text=self._label_for(value))
        if self.command:
            self.command(value)
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()


class EdgeTTS:
    def __init__(self):
        self.voices = []
        self.last_error = ""
        self.load_voices()

    async def fetch_voices(self):
        return await asyncio.wait_for(edge_tts.list_voices(), timeout=8)

    def load_voices(self):
        try:
            self.voices = asyncio.run(self.fetch_voices())
            self.last_error = ""
            print(f"Loaded {len(self.voices)} Microsoft Edge Neural voices.")
            return True
        except Exception as error:
            self.voices = []
            self.last_error = str(error)
            print(f"Microsoft Edge Neural voice catalog unavailable: {self.last_error}")
            return False

    def reload_voices(self):
        return self.load_voices()

    def get_languages(self):
        return sorted({voice["Locale"] for voice in self.voices})

    def get_language_name(self, language):
        return next((v["LocaleName"] for v in self.get_voices_for_language(language) if v.get("LocaleName")), language)

    def get_language_items(self):
        return [{"value": language, "display": self.get_language_name(language), "gender": "Unknown"} for language in self.get_languages()]

    def get_voices_for_language(self, language):
        return [voice for voice in self.voices if voice["Locale"].lower() == language.lower()]

    def get_default_voice(self, language):
        voices = self.get_voices_for_language(language)
        return voices[0]["ShortName"] if voices else ""

    async def synthesize(self, text, voice, rate, pitch, output_path):
        rate_value = f"{rate:+d}%"
        pitch_value = f"{pitch:+d}Hz"
        communicator = edge_tts.Communicate(text, voice, rate=rate_value, pitch=pitch_value)
        await communicator.save(output_path)

    def save_mp3(self, text, voice, rate, pitch, output_path):
        asyncio.run(self.synthesize(text, voice, rate, pitch, output_path))


class WindowsSAPI:
    def __init__(self):
        self.process = None
        self.voices = []
        self.load_voices()

    def load_voices(self):
        self.voices = []
        if platform.system() != "Windows":
            return
        command = "Add-Type -AssemblyName System.Speech\n$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer\nforeach ($voice in $synth.GetInstalledVoices()) {$info = $voice.VoiceInfo; Write-Output \"$($info.Name)|$($info.Culture.Name)|$($info.Gender)|$($info.Culture.DisplayName)\"}"
        try:
            result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    self.voices.append({"name": parts[0], "language": parts[1], "gender": parts[2], "language_name": parts[3], "id": parts[0]})
        except Exception:
            self.voices = []

    def get_languages(self):
        return sorted({voice["language"] for voice in self.voices})

    def get_language_name(self, language):
        return next((v["language_name"] for v in self.get_voices_for_language(language) if v.get("language_name")), language)

    def get_language_items(self):
        return [{"value": language, "display": self.get_language_name(language), "gender": "Unknown"} for language in self.get_languages()]

    def get_voices_for_language(self, language):
        return [voice for voice in self.voices if voice["language"].lower() == language.lower()]

    def speak(self, text, voice, rate):
        self.stop()
        safe_text = text.replace("'", "''")
        safe_voice = voice.replace("'", "''")
        sapi_rate = max(-10, min(10, int(rate / 10)))
        voice_code = f"$speak.SelectVoice('{safe_voice}')" if voice else ""
        command = f"Add-Type -AssemblyName System.Speech\n$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer\n$speak.Rate = {sapi_rate}\n{voice_code}\n$speak.Speak('{safe_text}')"
        self.process = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    def is_speaking(self):
        if self.process is None:
            return False
        if self.process.poll() is None:
            return True
        self.process = None
        return False


class TTSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.edge_tts = EdgeTTS()
        self.sapi = WindowsSAPI()
        self.engine = DEFAULT_ENGINE
        self.is_speaking_now = False
        self.placeholder_active = True
        self.edge_process = None
        self.playback_token = 0
        self.title("VoiceMDX TTS Editor")
        self.geometry("900x700")
        self.resizable(False, False)
        self.center_window()
        self.initialize_audio()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.create_ui()
        self.setup_placeholder()
        self.setup_defaults()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def initialize_audio(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except Exception:
            pass

    def center_window(self):
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = max(0, (screen_width - 900) // 2)
        y = max(0, (screen_height - 700) // 2)
        self.geometry(f"900x700+{x}+{y}")

    def create_ui(self):
        title_frame = ctk.CTkFrame(self, fg_color="transparent", border_width=0)
        title_frame.grid(row=0, column=0, padx=30, pady=(25, 10), sticky="ew")
        title_frame.grid_columnconfigure(0, weight=1)
        title_label = ctk.CTkLabel(title_frame, text="Text to Speech", font=ctk.CTkFont(size=32, weight="bold"))
        title_label.grid(row=0, column=0, sticky="w")
        self.system_label = ctk.CTkLabel(title_frame, text="", text_color="gray")
        self.system_label.grid(row=0, column=1, sticky="e")
        self.textbox = ctk.CTkTextbox(self, font=ctk.CTkFont(size=18), corner_radius=12, border_width=1, wrap="word")
        self.textbox.grid(row=1, column=0, padx=30, pady=10, sticky="nsew")
        settings = ctk.CTkFrame(self)
        settings.grid(row=2, column=0, padx=30, pady=15, sticky="ew")
        settings.grid_columnconfigure(1, weight=1)
        engine_label = ctk.CTkLabel(settings, text="Engine")
        engine_label.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        self.engine_combo = ctk.CTkComboBox(settings, values=["Microsoft Edge Neural", "Windows SAPI"], height=38, command=self.engine_changed)
        self.engine_combo.grid(row=0, column=1, padx=15, pady=(15, 8), sticky="ew")
        self.engine_combo.set(DEFAULT_ENGINE)
        language_label = ctk.CTkLabel(settings, text="Language")
        language_label.grid(row=1, column=0, padx=15, pady=8, sticky="w")
        self.language_selector = GridSelector(settings, width=500, height=38, columns=3, command=self.language_changed)
        self.language_selector.grid(row=1, column=1, padx=15, pady=8, sticky="ew")
        voice_label = ctk.CTkLabel(settings, text="Voice")
        voice_label.grid(row=2, column=0, padx=15, pady=8, sticky="w")
        self.voice_selector = GridSelector(settings, width=500, height=38, columns=3, gender_grouped=True)
        self.voice_selector.grid(row=2, column=1, padx=15, pady=8, sticky="ew")
        rate_label = ctk.CTkLabel(settings, text="Speed")
        rate_label.grid(row=3, column=0, padx=15, pady=8, sticky="w")
        rate_frame = ctk.CTkFrame(settings, fg_color="transparent", border_width=0)
        rate_frame.grid(row=3, column=1, padx=15, pady=8, sticky="ew")
        rate_frame.grid_columnconfigure(0, weight=1)
        self.rate_slider = ctk.CTkSlider(rate_frame, from_=-50, to=100, number_of_steps=150, border_width=0, button_length=18, height=18, corner_radius=1000, command=self.update_rate)
        self.rate_slider.grid(row=0, column=0, sticky="ew")
        self.rate_slider.set(0)
        self.rate_value = ctk.CTkLabel(rate_frame, text="Normal", width=55)
        self.rate_value.grid(row=0, column=1, padx=(12, 0))
        pitch_label = ctk.CTkLabel(settings, text="Pitch")
        pitch_label.grid(row=4, column=0, padx=15, pady=(8, 15), sticky="w")
        pitch_frame = ctk.CTkFrame(settings, fg_color="transparent", border_width=0)
        pitch_frame.grid(row=4, column=1, padx=15, pady=(8, 15), sticky="ew")
        pitch_frame.grid_columnconfigure(0, weight=1)
        self.pitch_slider = ctk.CTkSlider(pitch_frame, from_=-50, to=50, number_of_steps=100, border_width=0, button_length=18, height=18, corner_radius=1000, command=self.update_pitch)
        self.pitch_slider.grid(row=0, column=0, sticky="ew")
        self.pitch_slider.set(0)
        self.pitch_value = ctk.CTkLabel(pitch_frame, text="Normal", width=55)
        self.pitch_value.grid(row=0, column=1, padx=(12, 0))
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent", border_width=0)
        bottom_frame.grid(row=3, column=0, padx=30, pady=(0, 15), sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(bottom_frame, text="Ready", text_color="gray")
        self.status_label.grid(row=0, column=0, sticky="w")
        save_button = ctk.CTkButton(bottom_frame, text="Save as WAV", width=160, height=48, command=self.save_wav)
        save_button.grid(row=0, column=1, padx=(15, 8))
        self.speak_button = ctk.CTkButton(bottom_frame, text="🔊 Speak", width=180, height=48, font=ctk.CTkFont(size=17, weight="bold"), command=self.toggle_speech)
        self.speak_button.grid(row=0, column=2, padx=(8, 0))

    def setup_defaults(self):
        if self.edge_tts.voices:
            self.use_edge_engine()
        else:
            self.engine_combo.set("Windows SAPI")
            self.use_sapi_engine("Microsoft Edge Neural unavailable • Using Windows SAPI")

    def setup_placeholder(self):
        self.textbox.insert("1.0", PLACEHOLDER_TEXT)
        self.textbox.configure(text_color="gray")
        self.textbox.bind("<FocusIn>", self.remove_placeholder)
        self.textbox.bind("<FocusOut>", self.restore_placeholder)

    def remove_placeholder(self, event=None):
        if self.placeholder_active:
            self.textbox.delete("1.0", "end")
            self.textbox.configure(text_color=ctk.ThemeManager.theme["CTkTextbox"]["text_color"])
            self.placeholder_active = False

    def restore_placeholder(self, event=None):
        if not self.textbox.get("1.0", "end").strip():
            self.textbox.insert("1.0", PLACEHOLDER_TEXT)
            self.textbox.configure(text_color="gray")
            self.placeholder_active = True

    def update_rate(self, value):
        value = int(value)
        self.rate_value.configure(text="Normal" if value == 0 else f"{value:+d}%")

    def update_pitch(self, value):
        value = int(value)
        self.pitch_value.configure(text="Normal" if value == 0 else f"{value:+d}%")

    def normalize_gender(self, gender):
        gender_text = str(gender or "").strip().lower()
        if gender_text == "female":
            return "Female"
        if gender_text == "male":
            return "Male"
        if gender_text:
            return gender_text.title()
        return "Unknown"

    def apply_language_items(self, items):
        self.language_selector.configure(items=items)
        languages = [item["value"] for item in items]
        selected = DEFAULT_LANGUAGE if DEFAULT_LANGUAGE in languages else (items[0]["value"] if items else "")
        self.language_selector.set(selected)
        self.language_changed(selected)

    def use_edge_engine(self):
        self.engine = "Microsoft Edge Neural"
        self.apply_language_items(self.edge_tts.get_language_items())
        self.system_label.configure(text=f"Microsoft Edge Neural • {len(self.edge_tts.voices)} voices")
        self.status_label.configure(text="Microsoft Edge Neural ready")

    def use_sapi_engine(self, message="Windows SAPI ready"):
        self.engine = "Windows SAPI"
        self.apply_language_items(self.sapi.get_language_items())
        self.system_label.configure(text=f"Windows SAPI • {len(self.sapi.voices)} voices")
        self.status_label.configure(text=message)

    def engine_changed(self, engine):
        if engine == "Microsoft Edge Neural":
            if self.edge_tts.voices:
                self.use_edge_engine()
                return
            self.status_label.configure(text="Connecting to Microsoft Edge Neural...")
            self.update_idletasks()
            if not self.edge_tts.reload_voices():
                self.engine_combo.set("Windows SAPI")
                self.use_sapi_engine("Microsoft Edge Neural unavailable • Using Windows SAPI")
                return
            self.use_edge_engine()
        else:
            self.use_sapi_engine()

    def build_voice_items(self, language):
        if self.engine == "Microsoft Edge Neural":
            return [{"value": v["ShortName"], "display": v["ShortName"], "gender": self.normalize_gender(v.get("Gender"))} for v in self.edge_tts.get_voices_for_language(language)]
        return [{"value": v["name"], "display": v["name"], "gender": self.normalize_gender(v.get("gender"))} for v in self.sapi.get_voices_for_language(language)]

    def language_changed(self, language):
        if not language:
            self.voice_selector.configure(items=[])
            self.voice_selector.set("")
            return
        items = self.build_voice_items(language)
        self.voice_selector.configure(items=items)
        self.voice_selector.set(items[0]["value"] if items else "")

    def get_text(self):
        text = self.textbox.get("1.0", "end").strip()
        return PLACEHOLDER_TEXT if self.placeholder_active or not text else text

    def get_voice(self):
        return self.voice_selector.get()

    def toggle_speech(self):
        if self.is_speaking_now:
            self.stop_speech()
        else:
            self.start_speech()

    def start_speech(self):
        text = self.get_text()
        language = self.language_selector.get()
        voice = self.get_voice()
        rate = int(self.rate_slider.get())
        pitch = int(self.pitch_slider.get())
        if not voice:
            self.status_label.configure(text=f"No voice available for {language}.")
            return
        self.is_speaking_now = True
        self.playback_token += 1
        current_token = self.playback_token
        self.speak_button.configure(text="⏹ Stop")
        self.status_label.configure(text=f"Speaking • {voice}")

        def worker():
            try:
                if self.engine == "Microsoft Edge Neural":
                    temp_path = Path(tempfile.gettempdir()) / f"voicemdx_tts_{current_token}.mp3"
                    self.edge_tts.save_mp3(text, voice, rate, pitch, str(temp_path))
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    pygame.mixer.music.load(str(temp_path))
                    pygame.mixer.music.play()
                    while self.is_speaking_now and self.playback_token == current_token and pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                    pygame.mixer.music.stop()
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                else:
                    self.sapi.speak(text, voice, rate)
                    while self.is_speaking_now and self.playback_token == current_token and self.sapi.is_speaking():
                        time.sleep(0.05)
                if self.is_speaking_now and self.playback_token == current_token:
                    self.after(0, self.speech_finished)
            except Exception as error:
                error_message = str(error)
                self.after(0, lambda error_message=error_message: self.speech_error(error_message))

        threading.Thread(target=worker, daemon=True).start()

    def speech_finished(self):
        if self.is_speaking_now:
            self.is_speaking_now = False
            self.speak_button.configure(text="🔊 Speak")
            self.status_label.configure(text="Ready")

    def speech_error(self, error):
        self.is_speaking_now = False
        self.speak_button.configure(text="🔊 Speak")
        self.status_label.configure(text=f"TTS Error: {error}")

    def stop_speech(self):
        self.playback_token += 1
        if self.engine == "Windows SAPI":
            self.sapi.stop()
        else:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        self.is_speaking_now = False
        self.speak_button.configure(text="🔊 Speak")
        self.status_label.configure(text="Stopped")

    def save_wav(self):
        text = self.get_text()
        language = self.language_selector.get()
        voice = self.get_voice()
        rate = int(self.rate_slider.get())
        pitch = int(self.pitch_slider.get())
        if not voice:
            self.status_label.configure(text=f"No voice available for {language}.")
            return
        output_path = filedialog.asksaveasfilename(title="Save Voice as WAV", defaultextension=".wav", filetypes=[("WAV audio", "*.wav")])
        if not output_path:
            return
        self.status_label.configure(text="Saving WAV...")

        def worker():
            try:
                if self.engine != "Microsoft Edge Neural":
                    raise RuntimeError("WAV export is currently supported through Microsoft Edge Neural.")
                temp_mp3 = Path(tempfile.gettempdir()) / "voicemdx_export.mp3"
                self.edge_tts.save_mp3(text, voice, rate, pitch, str(temp_mp3))
                ffmpeg = shutil.which("ffmpeg")
                if not ffmpeg:
                    raise RuntimeError("FFmpeg is required for WAV export. Install FFmpeg and add it to PATH.")
                subprocess.run(["ffmpeg", "-y", "-i", str(temp_mp3), str(output_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    temp_mp3.unlink()
                except Exception:
                    pass
                self.after(0, self.wav_finished)
            except Exception as error:
                error_message = str(error)
                self.after(0, lambda error_message=error_message: self.wav_error(error_message))

        threading.Thread(target=worker, daemon=True).start()

    def wav_finished(self):
        self.status_label.configure(text="WAV saved successfully")

    def wav_error(self, error):
        self.status_label.configure(text=f"WAV Error: {error}")

    def on_close(self):
        self.is_speaking_now = False
        self.playback_token += 1
        self.sapi.stop()
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception:
            pass
        if self.edge_process is not None:
            try:
                self.edge_process.terminate()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(str(BASE_DIR / "themes" / "red.json"))
    app = TTSApp()
    app.mainloop()