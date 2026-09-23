import customtkinter as ctk
import asyncio
import edge_tts
import subprocess
import threading
import platform
import os
import shutil
import tempfile
import time
import sys
import pygame
from pathlib import Path
from tkinter import filedialog
from utils.centwin import center_window
from utils.ctk_theme import configure_ctk_theme
configure_ctk_theme()

DEFAULT_LANGUAGE = "en-US"
DEFAULT_ENGINE = "Microsoft Edge Neural"
PLACEHOLDER_TEXT = "Type something here..."
SELECTOR_POPUP_HEIGHT = 700
TAG_POPUP_HEIGHT = 500

class MultiTagSelector:
    def __init__(self, master, title, selected=None, options=None, command=None, width=600):
        self.master = master
        self.title = title
        self.selected = set(selected or [])
        self.options = sorted(set(options or []), key=str.lower)
        self.command = command
        self.popup = None
        self.variables = {}
        self.button = ctk.CTkButton(master, text=title, height=36, width=width, anchor="w", command=self.open)

    def grid(self, **kwargs):
        self.button.grid(**kwargs)

    def set_options(self, options):
        self.options = sorted(set(options or []), key=str.lower)
        self.selected.intersection_update(self.options)
        self.update_button()

    def update_button(self):
        if not self.selected:
            text = self.title
        else:
            selected = sorted(self.selected, key=str.lower)
            text = " · ".join(selected[:2])
            if len(selected) > 2:
                text = f"{text} +{len(selected) - 2}"
        self.button.configure(text=text)

    def open(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = ctk.CTkToplevel(self.master.winfo_toplevel())
        self.popup.title("Select Tags")
        self.popup.transient(self.master.winfo_toplevel())
        self.popup.resizable(False, False)
        width = 560
        height = TAG_POPUP_HEIGHT
        center_window(self.popup, width=width, height=height)
        self.popup.grid_rowconfigure(1, weight=1)
        self.popup.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(self.popup, text=self.title, font=ctk.CTkFont(size=16, weight="bold"))
        title.grid(row=0, column=0, padx=16, pady=(16, 10), sticky="w")
        frame = ctk.CTkScrollableFrame(self.popup, width=width - 32, height=height - 120, corner_radius=10)
        frame.grid(row=1, column=0, padx=16, pady=(0, 10), sticky="nsew")
        self.variables = {}
        for index, option in enumerate(self.options):
            variable = ctk.BooleanVar(value=option in self.selected)
            checkbox = ctk.CTkCheckBox(frame, text=option, variable=variable, command=lambda value=option: self.toggle(value))
            checkbox.grid(row=index // 2, column=index % 2, padx=8, pady=6, sticky="w")
            self.variables[option] = variable
        if not self.options:
            label = ctk.CTkLabel(frame, text="No tags available.", text_color="gray")
            label.grid(row=0, column=0, padx=10, pady=20)
        buttons = ctk.CTkFrame(self.popup, fg_color="transparent", border_width=0)
        buttons.grid(row=2, column=0, padx=16, pady=(0, 14), sticky="e")
        clear_button = ctk.CTkButton(buttons, text="Clear", width=90, height=34, command=self.clear)
        clear_button.grid(row=0, column=0, padx=(0, 8))
        done_button = ctk.CTkButton(buttons, text="Done", width=90, height=34, command=self.close)
        done_button.grid(row=0, column=1)

    def toggle(self, value):
        variable = self.variables.get(value)
        if variable is not None and variable.get():
            self.selected.add(value)
        else:
            self.selected.discard(value)
        self.update_button()
        if self.command:
            self.command(set(self.selected))

    def clear(self):
        self.selected.clear()
        self.update_button()
        for variable in self.variables.values():
            variable.set(False)
        if self.command:
            self.command(set())

    def close(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()

class GridSelector:
    def __init__(self, master, values=None, command=None, width=400, height=38, columns=4, gender_grouped=False, compact=False):
        self.master = master
        self.values = values or []
        self.items = []
        self.command = command
        self.width = width
        self.height = height
        self.columns = columns
        self.gender_grouped = gender_grouped
        self.compact = compact
        self.content_options = []
        self.personality_options = []
        self.content_selected = set()
        self.personality_selected = set()
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
            self.content_options = sorted({tag for item in self.items for tag in item.get("content_categories", [])}, key=str.lower)
            self.personality_options = sorted({tag for item in self.items for tag in item.get("voice_personalities", [])}, key=str.lower)
            self.content_selected.intersection_update(self.content_options)
            self.personality_selected.intersection_update(self.personality_options)
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
        popup_width = 1260 if self.compact else 900
        popup_height = SELECTOR_POPUP_HEIGHT
        self.center_popup(popup_width, popup_height)
        self.popup.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(self.popup, text="Select", font=ctk.CTkFont(size=19, weight="bold"))
        title.grid(row=0, column=0, padx=18, pady=(16, 10), sticky="w")
        search_frame = ctk.CTkFrame(self.popup, fg_color="transparent", border_width=0)
        search_frame.grid(row=1, column=0, padx=18, pady=(0, 10), sticky="ew")
        search_frame.grid_columnconfigure(0, weight=1)
        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search...", height=40, font=ctk.CTkFont(size=14))
        self.search_entry.grid(row=0, column=0, sticky="ew")
        self.search_entry.bind("<KeyRelease>", self.filter_items)
        results_row = 2
        results_height = popup_height - 115
        if self.compact:
            tag_frame = ctk.CTkFrame(self.popup, fg_color="transparent", border_width=0)
            tag_frame.grid(row=2, column=0, padx=18, pady=(0, 10), sticky="ew")
            tag_frame.grid_columnconfigure(0, weight=1)
            tag_frame.grid_columnconfigure(1, weight=1)
            content_filter = MultiTagSelector(tag_frame, "All Content Categories", self.content_selected, self.content_options, lambda selected: self.tag_filter_changed("content", selected), width=(popup_width - 55) // 2)
            content_filter.grid(row=0, column=0, padx=(0, 5), sticky="ew")
            personality_filter = MultiTagSelector(tag_frame, "All Voice Personalities", self.personality_selected, self.personality_options, lambda selected: self.tag_filter_changed("personality", selected), width=(popup_width - 55) // 2)
            personality_filter.grid(row=0, column=1, padx=(5, 0), sticky="ew")
            results_row = 3
            results_height = popup_height - 165
        self.popup.grid_rowconfigure(results_row, weight=1)
        self.scroll_frame = ctk.CTkScrollableFrame(self.popup, width=popup_width - 36, height=results_height, corner_radius=12)
        self.scroll_frame.grid(row=results_row, column=0, padx=18, pady=(0, 18), sticky="nsew")
        self.configure_columns()
        self.render_items()
        self.popup.after_idle(self.refresh_scroll_region)

    def center_popup(self, width, height):
        center_window(self.popup, width=width, height=height)

    def refresh_scroll_region(self):
        if self.scroll_frame is None or not self.scroll_frame.winfo_exists():
            return
        self.scroll_frame.update_idletasks()

    def filter_items(self, event=None):
        self.render_items()

    def get_filtered_items(self):
        search_text = self.search_entry.get().strip().lower() if self.search_entry is not None else ""
        items = self.items
        if search_text:
            items = [item for item in items if search_text in item["value"].lower() or search_text in item["display"].lower() or search_text in item.get("gender", "").lower() or search_text in " ".join(item.get("content_categories", [])).lower() or search_text in " ".join(item.get("voice_personalities", [])).lower()]
        if self.content_selected:
            items = [item for item in items if any(tag in item.get("content_categories", []) for tag in self.content_selected)]
        if self.personality_selected:
            items = [item for item in items if any(tag in item.get("voice_personalities", []) for tag in self.personality_selected)]
        return items

    def tag_filter_changed(self, kind, selected):
        if kind == "content":
            self.content_selected = set(selected)
        else:
            self.personality_selected = set(selected)
        self.render_items()

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

    def set_card_hover(self, card, active):
        card.configure(border_width=2 if active else 1)

    def bind_card_hover(self, card, *widgets):
        for widget in (card, *widgets):
            widget.bind("<Enter>", lambda event, target=card: self.set_card_hover(target, True))
            widget.bind("<Leave>", lambda event, target=card: self.set_card_hover(target, False))

    def create_button(self, item):
        if not self.compact:
            return ctk.CTkButton(self.scroll_frame, text=item["display"], height=40, command=lambda value=item["value"]: self.select(value))
        card = ctk.CTkFrame(self.scroll_frame, height=94, corner_radius=10, border_width=1)
        card.grid_propagate(False)
        name_label = ctk.CTkLabel(card, text=item["display"], anchor="w", font=ctk.CTkFont(size=16, weight="bold"))
        name_label.pack(fill="x", padx=13, pady=(10, 3))
        categories = " · ".join(item.get("content_categories", []))
        personalities = " · ".join(item.get("voice_personalities", []))
        category_label = ctk.CTkLabel(card, text=categories or "No category", anchor="w", text_color="gray", font=ctk.CTkFont(size=12))
        category_label.pack(fill="x", padx=13, pady=(0, 3))
        personality_label = ctk.CTkLabel(card, text=personalities or "No personality", anchor="w", text_color="gray", font=ctk.CTkFont(size=12))
        personality_label.pack(fill="x", padx=13, pady=(0, 8))
        select_command = lambda event=None, value=item["value"]: self.select(value)
        for widget in (card, name_label, category_label, personality_label):
            widget.bind("<Button-1>", select_command)
        self.bind_card_hover(card, name_label, category_label, personality_label)
        return card

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
    def __init__(self, apply_on_source_path=None):
        super().__init__()
        self.apply_on_source_path = Path(apply_on_source_path) if apply_on_source_path else None
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
        self.after(100, self.bring_to_front)

    def bring_to_front(self):
        try:
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)
            self.focus_force()
            if platform.system() == "Windows":
                import ctypes
                hwnd = self.winfo_id()
                ctypes.windll.user32.ShowWindow(hwnd, 5)
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            self.after(500, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

    def initialize_audio(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except Exception:
            pass

    def center_window(self):
        center_window(self, width=900, height=700)

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
        settings.grid_columnconfigure(3, weight=1)
        engine_label = ctk.CTkLabel(settings, text="ENGINE")
        engine_label.grid(row=0, column=0, padx=(18, 10), pady=(15, 8), sticky="w")
        self.engine_combo = ctk.CTkOptionMenu(settings, values=["Microsoft Edge Neural", "Windows SAPI"], height=38, command=self.engine_changed)
        self.engine_combo.grid(row=0, column=1, columnspan=3, padx=(0, 18), pady=(15, 8), sticky="ew")
        language_label = ctk.CTkLabel(settings, text="LANGUAGE")
        language_label.grid(row=1, column=0, padx=(18, 10), pady=8, sticky="w")
        self.language_selector = GridSelector(settings, width=300, height=38, columns=3, command=self.language_changed)
        self.language_selector.grid(row=1, column=1, padx=(0, 18), pady=8, sticky="ew")
        country_label = ctk.CTkLabel(settings, text="COUNTRY")
        country_label.grid(row=1, column=2, padx=(18, 10), pady=8, sticky="w")
        self.country_selector = GridSelector(settings, width=300, height=38, columns=3, command=self.country_changed)
        self.country_selector.grid(row=1, column=3, padx=(0, 18), pady=8, sticky="ew")
        gender_label = ctk.CTkLabel(settings, text="GENDER")
        gender_label.grid(row=2, column=0, padx=(18, 10), pady=8, sticky="w")
        self.gender_combo = ctk.CTkOptionMenu(settings, values=["All"], height=38, command=self.gender_changed)
        self.gender_combo.grid(row=2, column=1, padx=(0, 18), pady=8, sticky="ew")
        self.gender_combo.set("All")
        voice_label = ctk.CTkLabel(settings, text="VOICE")
        voice_label.grid(row=2, column=2, padx=(18, 10), pady=8, sticky="w")
        self.voice_selector = GridSelector(settings, width=300, height=38, columns=3, compact=True)
        self.voice_selector.grid(row=2, column=3, padx=(0, 18), pady=8, sticky="ew")
        rate_label = ctk.CTkLabel(settings, text="Speed")
        rate_label.grid(row=3, column=0, padx=(18, 10), pady=8, sticky="w")
        rate_frame = ctk.CTkFrame(settings, fg_color="transparent", border_width=0)
        rate_frame.grid(row=3, column=1, columnspan=3, padx=(0, 18), pady=8, sticky="ew")
        rate_frame.grid_columnconfigure(0, weight=1)
        self.rate_slider = ctk.CTkSlider(rate_frame, from_=-50, to=100, number_of_steps=150, border_width=0, button_length=18, height=18, corner_radius=1000, command=self.update_rate)
        self.rate_slider.grid(row=0, column=0, sticky="ew")
        self.rate_slider.set(0)
        self.rate_value = ctk.CTkLabel(rate_frame, text="Normal", width=55)
        self.rate_value.grid(row=0, column=1, padx=(12, 0))
        pitch_label = ctk.CTkLabel(settings, text="Pitch")
        pitch_label.grid(row=4, column=0, padx=(18, 10), pady=(8, 15), sticky="w")
        pitch_frame = ctk.CTkFrame(settings, fg_color="transparent", border_width=0)
        pitch_frame.grid(row=4, column=1, columnspan=3, padx=(0, 18), pady=(8, 15), sticky="ew")
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
        if self.apply_on_source_path is not None:
            self.apply_button = ctk.CTkButton(bottom_frame, text="Apply on Source", width=180, height=48, font=ctk.CTkFont(size=17, weight="bold"), command=self.apply_on_source)
            self.apply_button.grid(row=0, column=3, padx=(8, 0))

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
        language_map = {}
        default_language = ""
        for item in items:
            language = item["language"]
            language_map.setdefault(language, []).append(item)
            if item["value"] == DEFAULT_LANGUAGE:
                default_language = language
        self.locale_items = items
        self.language_map = language_map
        languages = list(language_map)
        selected = default_language or (languages[0] if languages else "")
        self.language_selector.configure(items=[{"value": language, "display": language, "gender": "Unknown"} for language in languages])
        self.language_selector.set(selected)
        self.language_changed(selected)

    def use_edge_engine(self):
        self.engine = "Microsoft Edge Neural"
        self.apply_language_items(self.get_locale_items("Microsoft Edge Neural"))
        self.system_label.configure(text=f"Microsoft Edge Neural • {len(self.edge_tts.voices)} voices")
        self.status_label.configure(text="Microsoft Edge Neural ready")

    def use_sapi_engine(self, message="Windows SAPI ready"):
        self.engine = "Windows SAPI"
        self.apply_language_items(self.get_locale_items("Windows SAPI"))
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

    def get_locale_items(self, engine):
        source = self.edge_tts.voices if engine == "Microsoft Edge Neural" else self.sapi.voices
        items = []
        seen = set()
        for voice in source:
            locale = voice["Locale"] if engine == "Microsoft Edge Neural" else voice["language"]
            if locale in seen:
                continue
            seen.add(locale)
            full_name = self.edge_tts.get_language_name(locale) if engine == "Microsoft Edge Neural" else self.sapi.get_language_name(locale)
            language_name = full_name.split("(", 1)[0].strip()
            country_name = full_name.split("(", 1)[1].rsplit(")", 1)[0].strip() if "(" in full_name and ")" in full_name else locale.split("-", 1)[1] if "-" in locale else locale
            items.append({"value": locale, "display": language_name, "language": language_name, "country": country_name})
        return sorted(items, key=lambda item: (item["language"].lower(), item["country"].lower()))

    def build_voice_items(self, locale):
        if self.engine == "Microsoft Edge Neural":
            voices = self.edge_tts.get_voices_for_language(locale)
            return [{"value": v["ShortName"], "display": v["ShortName"], "gender": self.normalize_gender(v.get("Gender")), "content_categories": v.get("VoiceTag", {}).get("ContentCategories", []) or [], "voice_personalities": v.get("VoiceTag", {}).get("VoicePersonalities", []) or []} for v in voices]
        voices = self.sapi.get_voices_for_language(locale)
        return [{"value": v["name"], "display": v["name"], "gender": self.normalize_gender(v.get("gender")), "content_categories": [], "voice_personalities": []} for v in voices]

    def language_changed(self, language):
        locales = self.language_map.get(language, [])
        country_items = [{"value": item["value"], "display": item["country"], "gender": "Unknown"} for item in locales]
        self.country_selector.configure(items=country_items)
        if not locales:
            self.selected_locale = ""
            self.country_selector.set("")
            self.voice_selector.configure(items=[])
            self.voice_selector.set("")
            self.update_gender_items([])
            return
        selected = next((item["value"] for item in locales if item["value"] == getattr(self, "selected_locale", "")), locales[0]["value"])
        if language == "English":
            selected = next((item["value"] for item in locales if item["value"] == DEFAULT_LANGUAGE), selected)
        self.selected_locale = selected
        self.country_selector.set(selected)
        self.refresh_voices()

    def country_changed(self, locale):
        if not locale:
            return
        item = next((item for item in self.locale_items if item["value"] == locale), None)
        if item is None:
            return
        self.selected_locale = locale
        self.language_selector.set(item["language"])
        self.refresh_voices()

    def gender_changed(self, gender):
        self.refresh_voices()

    def refresh_voices(self):
        locale = getattr(self, "selected_locale", "")
        all_items = self.build_voice_items(locale) if locale else []
        self.update_gender_items(all_items)
        selected_gender = self.gender_combo.get()
        items = all_items if selected_gender == "All" else [item for item in all_items if item["gender"] == selected_gender]
        self.voice_selector.configure(items=items)
        current_voice = self.voice_selector.get()
        selected_voice = current_voice if any(item["value"] == current_voice for item in items) else (items[0]["value"] if items else "")
        self.voice_selector.set(selected_voice)

    def update_gender_items(self, items):
        genders = sorted({item["gender"] for item in items if item["gender"] not in ("Unknown", "")})
        values = ["All"] + genders
        current = self.gender_combo.get()
        self.gender_combo.configure(values=values)
        self.gender_combo.set(current if current in values else "All")

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

    def apply_on_source(self):
        if self.is_speaking_now:
            self.stop_speech()
        text = self.get_text()
        language = self.language_selector.get()
        voice = self.get_voice()
        rate = int(self.rate_slider.get())
        pitch = int(self.pitch_slider.get())
        if not voice:
            self.status_label.configure(text=f"No voice available for {language}.")
            return
        if self.engine != "Microsoft Edge Neural":
            self.status_label.configure(text="Apply on Source requires Microsoft Edge Neural.")
            return
        if self.apply_on_source_path is None:
            self.status_label.configure(text="Apply on Source is unavailable.")
            return
        self.apply_button.configure(state="disabled")
        self.status_label.configure(text="Creating source WAV...")

        def worker():
            temp_mp3 = Path(tempfile.gettempdir()) / f"voicemdx_apply_{os.getpid()}_{int(time.time() * 1000)}.mp3"
            try:
                self.edge_tts.save_mp3(text, voice, rate, pitch, str(temp_mp3))
                ffmpeg = shutil.which("ffmpeg")
                if not ffmpeg:
                    raise RuntimeError("FFmpeg is required for WAV export. Install FFmpeg and add it to PATH.")
                self.apply_on_source_path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([ffmpeg, "-y", "-i", str(temp_mp3), "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(self.apply_on_source_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    temp_mp3.unlink()
                except Exception:
                    pass
                self.after(0, self.apply_on_source_finished)
            except Exception as error:
                try:
                    temp_mp3.unlink()
                except Exception:
                    pass
                error_message = str(error)
                self.after(0, lambda error_message=error_message: self.apply_on_source_error(error_message))

        threading.Thread(target=worker, daemon=True).start()

    def apply_on_source_finished(self):
        self.status_label.configure(text="Source WAV applied")
        self.on_close()

    def apply_on_source_error(self, error):
        self.apply_button.configure(state="normal")
        self.status_label.configure(text=f"Apply Error: {error}")

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
    apply_on_source_path = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--apply-on-source":
        apply_on_source_path = sys.argv[2]
    app = TTSApp(apply_on_source_path=apply_on_source_path)
    app.mainloop()