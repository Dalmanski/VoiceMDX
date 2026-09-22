import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
import customtkinter as ctk
import numpy as np
import sounddevice as sd
import soundfile as sf
from tkinter import filedialog
from utils.centwin import center_window
from utils.ctk_theme import configure_ctk_theme

configure_ctk_theme()

class MicRecorderApp(ctk.CTk):
    def __init__(self, apply_on_source_path=None):
        super().__init__()
        self.apply_on_source_path = Path(apply_on_source_path) if apply_on_source_path else None
        self.title("Microphone Recorder")
        self.geometry("920x760")
        center_window(self, width=920, height=760)
        self.minsize(820, 700)
        self.recording = False
        self.paused = False
        self.recording_session = False
        self.stream = None
        self.audio_chunks = []
        self.preview_data = None
        self.preview_samplerate = None
        self.temp_path = None
        self.recorded_duration = 0.0
        self.record_segment_start = None
        self.volume_level = 0.0
        self.record_timer_job = None
        self.volume_timer_job = None
        self.preview_playing = False
        self.preview_started_at = None
        self.preview_position = 0.0
        self.preview_duration = 0.0
        self.preview_timer_job = None
        self.devices = {}
        self.build_ui()
        self.refresh_devices()

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        title = ctk.CTkLabel(self, text="Microphone Recorder", font=ctk.CTkFont(size=28, weight="bold"))
        title.grid(row=0, column=0, padx=35, pady=(25, 18), sticky="w")

        device_frame = ctk.CTkFrame(self)
        device_frame.grid(row=1, column=0, padx=35, pady=(0, 15), sticky="ew")
        device_frame.grid_columnconfigure(1, weight=1)

        device_label = ctk.CTkLabel(device_frame, text="MICROPHONE", font=ctk.CTkFont(size=13, weight="bold"))
        device_label.grid(row=0, column=0, padx=(20, 15), pady=18)

        self.device_menu = ctk.CTkOptionMenu(device_frame, values=["No active microphone found"], height=40)
        self.device_menu.grid(row=0, column=1, padx=10, pady=18, sticky="ew")

        self.refresh_button = ctk.CTkButton(device_frame, text="REFRESH", width=110, height=40, command=self.refresh_devices)
        self.refresh_button.grid(row=0, column=2, padx=(10, 20), pady=18)

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=2, column=0, padx=35, pady=(0, 15), sticky="nsew")
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        record_frame = ctk.CTkFrame(content)
        record_frame.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        record_frame.grid_columnconfigure(0, weight=1)

        record_header = ctk.CTkFrame(record_frame, fg_color="transparent")
        record_header.grid(row=0, column=0, padx=22, pady=(20, 8), sticky="ew")
        record_header.grid_columnconfigure(0, weight=1)

        record_title = ctk.CTkLabel(record_header, text="RECORDING", font=ctk.CTkFont(size=18, weight="bold"))
        record_title.grid(row=0, column=0, sticky="w")

        self.indicator_label = ctk.CTkLabel(record_header, text="● READY", font=ctk.CTkFont(size=13, weight="bold"), text_color="gray")
        self.indicator_label.grid(row=0, column=1, sticky="e")

        self.status_label = ctk.CTkLabel(record_frame, text="Ready to record", font=ctk.CTkFont(size=14))
        self.status_label.grid(row=1, column=0, padx=20, pady=(10, 0))

        self.timer_label = ctk.CTkLabel(record_frame, text="00:00", font=ctk.CTkFont(size=52, weight="bold"))
        self.timer_label.grid(row=2, column=0, padx=20, pady=(5, 25))

        controls = ctk.CTkFrame(record_frame, fg_color="transparent")
        controls.grid(row=3, column=0, padx=20, pady=(0, 20))

        self.record_button = ctk.CTkButton(controls, text="RECORD", width=160, height=44, command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=6)

        self.stop_button = ctk.CTkButton(controls, text="STOP", width=160, height=44, command=self.stop_recording, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=6)

        info_label = ctk.CTkLabel(record_frame, text="Click RECORD to start. Use PAUSE/PLAY to control the current recording.", font=ctk.CTkFont(size=12))
        info_label.grid(row=4, column=0, padx=20, pady=(0, 22))

        volume_frame = ctk.CTkFrame(content)
        volume_frame.grid(row=0, column=1, padx=(8, 0), sticky="nsew")

        volume_title = ctk.CTkLabel(volume_frame, text="INPUT VOLUME", font=ctk.CTkFont(size=15, weight="bold"))
        volume_title.pack(pady=(25, 10))

        self.volume_bar = ctk.CTkProgressBar(volume_frame, orientation="vertical", width=34, height=220)
        self.volume_bar.pack(pady=10)
        self.volume_bar.set(0)

        self.volume_label = ctk.CTkLabel(volume_frame, text="0%", font=ctk.CTkFont(size=14, weight="bold"))
        self.volume_label.pack(pady=(8, 2))

        volume_hint = ctk.CTkLabel(volume_frame, text="Live input level", font=ctk.CTkFont(size=12))
        volume_hint.pack(pady=(0, 20))

        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(row=3, column=0, padx=35, pady=(0, 25), sticky="ew")
        preview_frame.grid_columnconfigure(0, weight=1)

        preview_header = ctk.CTkFrame(preview_frame, fg_color="transparent")
        preview_header.grid(row=0, column=0, padx=22, pady=(18, 5), sticky="ew")
        preview_header.grid_columnconfigure(0, weight=1)

        preview_title = ctk.CTkLabel(preview_header, text="AUDIO PREVIEW", font=ctk.CTkFont(size=18, weight="bold"))
        preview_title.grid(row=0, column=0, sticky="w")

        self.preview_status_label = ctk.CTkLabel(preview_header, text="No recording", font=ctk.CTkFont(size=13))
        self.preview_status_label.grid(row=0, column=1, sticky="e")

        self.preview_label = ctk.CTkLabel(preview_frame, text="Your recorded audio will appear here after you stop recording.", font=ctk.CTkFont(size=13))
        self.preview_label.grid(row=1, column=0, padx=20, pady=(4, 12))

        self.preview_progress = ctk.CTkProgressBar(preview_frame, height=8)
        self.preview_progress.grid(row=2, column=0, padx=30, pady=(0, 8), sticky="ew")
        self.preview_progress.set(0)

        self.preview_timer_label = ctk.CTkLabel(preview_frame, text="00:00 / 00:00", font=ctk.CTkFont(size=18, weight="bold"))
        self.preview_timer_label.grid(row=3, column=0, padx=20, pady=5)

        preview_controls = ctk.CTkFrame(preview_frame, fg_color="transparent")
        preview_controls.grid(row=4, column=0, padx=20, pady=(8, 20))

        self.preview_button = ctk.CTkButton(preview_controls, text="PLAY", width=150, height=42, command=self.toggle_preview, state="disabled")
        self.preview_button.grid(row=0, column=0, padx=6)

        self.save_button = ctk.CTkButton(preview_controls, text="SAVE AS WAV", width=150, height=42, command=self.save_as_wav, state="disabled")
        self.save_button.grid(row=0, column=1, padx=6)

        self.apply_button = ctk.CTkButton(preview_controls, text="APPLY ON SOURCE", width=180, height=42, command=self.apply_source, state="normal" if self.apply_on_source_path else "disabled")
        self.apply_button.grid(row=0, column=2, padx=6)

    def get_wasapi_hostapi(self):
        for index, hostapi in enumerate(sd.query_hostapis()):
            if "WASAPI" in hostapi["name"].upper():
                return index
        return None

    def refresh_devices(self):
        try:
            wasapi_hostapi = self.get_wasapi_hostapi()
            self.devices = {}
            values = []
            if wasapi_hostapi is None:
                self.device_menu.configure(values=["Windows WASAPI not found"])
                self.device_menu.set("Windows WASAPI not found")
                self.status_label.configure(text="Windows WASAPI not found")
                return
            for index, device in enumerate(sd.query_devices()):
                if device["hostapi"] != wasapi_hostapi or device["max_input_channels"] <= 0:
                    continue
                name = device["name"].strip()
                if not name or "loopback" in name.lower() or name in self.devices:
                    continue
                try:
                    sd.check_input_settings(device=index, channels=1, dtype="float32")
                except Exception:
                    continue
                self.devices[name] = index
                values.append(name)
            values.sort(key=str.lower)
            if values:
                self.device_menu.configure(values=values)
                current = self.device_menu.get()
                self.device_menu.set(current if current in values else values[0])
                self.status_label.configure(text=f"{len(values)} active microphone(s) found")
            else:
                self.device_menu.configure(values=["No active microphone found"])
                self.device_menu.set("No active microphone found")
                self.status_label.configure(text="No active microphone found")
        except Exception as error:
            self.devices = {}
            self.device_menu.configure(values=["No active microphone found"])
            self.device_menu.set("No active microphone found")
            self.status_label.configure(text=f"Microphone scan error: {error}")

    def get_selected_device(self):
        return self.devices.get(self.device_menu.get())

    def toggle_recording(self):
        if not self.recording_session:
            self.start_recording()
        elif self.paused:
            self.resume_recording()
        else:
            self.pause_recording()

    def start_recording(self):
        device_index = self.get_selected_device()
        if device_index is None:
            self.status_label.configure(text="No active microphone selected")
            return
        try:
            device_info = sd.query_devices(device_index)
            samplerate = int(device_info["default_samplerate"])
            self.audio_chunks = []
            self.preview_data = None
            self.preview_samplerate = samplerate
            self.recorded_duration = 0.0
            self.record_segment_start = time.time()
            self.volume_level = 0.0
            self.stream = sd.InputStream(device=device_index, channels=1, samplerate=samplerate, dtype="float32", callback=self.audio_callback)
            self.stream.start()
            self.recording = True
            self.paused = False
            self.recording_session = True
            self.record_button.configure(text="PAUSE")
            self.stop_button.configure(state="normal")
            self.refresh_button.configure(state="disabled")
            self.device_menu.configure(state="disabled")
            self.preview_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            self.preview_status_label.configure(text="No recording")
            self.set_recording_indicator("● RECORDING", "green")
            self.status_label.configure(text=f"Recording from {device_info['name']}")
            self.timer_label.configure(text="00:00")
            self.update_record_timer()
            self.update_volume_meter()
        except Exception as error:
            self.status_label.configure(text=f"Recording error: {error}")

    def pause_recording(self):
        if not self.recording:
            return
        self.recorded_duration += time.time() - self.record_segment_start
        self.recording = False
        self.paused = True
        self.volume_level = 0.0
        self.record_button.configure(text="PLAY")
        self.set_recording_indicator("● PAUSED", "orange")
        self.status_label.configure(text="Recording paused")

    def resume_recording(self):
        if not self.recording_session or not self.paused:
            return
        self.record_segment_start = time.time()
        self.recording = True
        self.paused = False
        self.record_button.configure(text="PAUSE")
        self.set_recording_indicator("● RECORDING", "green")
        self.status_label.configure(text="Recording...")
        self.update_record_timer()
        self.update_volume_meter()

    def audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_chunks.append(indata.copy())
            rms = float(np.sqrt(np.mean(np.square(indata))))
            self.volume_level = min(1.0, rms * 5.0)
        else:
            self.volume_level = 0.0

    def stop_recording(self):
        if not self.recording_session:
            return
        if self.recording:
            self.recorded_duration += time.time() - self.record_segment_start
        self.recording = False
        self.paused = False
        self.recording_session = False
        self.volume_level = 0.0
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.record_button.configure(text="RECORD", state="normal")
        self.stop_button.configure(state="disabled")
        self.refresh_button.configure(state="normal")
        self.device_menu.configure(state="normal")
        self.timer_label.configure(text=self.format_duration(self.recorded_duration))
        self.cancel_record_timers()
        if not self.audio_chunks:
            self.set_recording_indicator("● READY", "gray")
            self.status_label.configure(text="No audio recorded")
            return
        self.preview_data = np.concatenate(self.audio_chunks, axis=0)
        self.preview_duration = len(self.preview_data) / self.preview_samplerate
        self.preview_position = 0.0
        try:
            if self.temp_path and os.path.exists(self.temp_path):
                os.remove(self.temp_path)
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            self.temp_path = temp_file.name
            temp_file.close()
            sf.write(self.temp_path, self.preview_data, self.preview_samplerate)
            self.preview_label.configure(text=f"Recording ready  •  {self.format_duration(self.preview_duration)}")
            self.preview_timer_label.configure(text=f"00:00 / {self.format_duration(self.preview_duration)}")
            self.preview_progress.set(0)
            self.preview_button.configure(state="normal", text="PLAY")
            self.save_button.configure(state="normal")
            self.preview_status_label.configure(text="Ready to preview")
            self.set_recording_indicator("● RECORDED", "gray")
            self.status_label.configure(text="Recording ready")
        except Exception as error:
            self.set_recording_indicator("● READY", "gray")
            self.status_label.configure(text=f"Temporary save error: {error}")

    def toggle_preview(self):
        if self.preview_playing:
            self.stop_preview()
        else:
            self.play_preview()

    def play_preview(self):
        if self.preview_data is None or self.preview_samplerate is None:
            return
        try:
            sd.stop()
            self.preview_position = 0.0
            self.preview_started_at = time.time()
            self.preview_playing = True
            sd.play(self.preview_data, self.preview_samplerate)
            self.preview_button.configure(text="STOP", state="normal")
            self.preview_status_label.configure(text="Playing")
            self.status_label.configure(text="Playing preview")
            self.update_preview_timer()
        except Exception as error:
            self.preview_playing = False
            self.preview_button.configure(text="PLAY")
            self.preview_status_label.configure(text="Ready to preview")
            self.status_label.configure(text=f"Playback error: {error}")

    def stop_preview(self):
        if not self.preview_playing:
            return
        self.preview_position = min(time.time() - self.preview_started_at, self.preview_duration)
        self.preview_playing = False
        sd.stop()
        self.cancel_preview_timer()
        self.preview_button.configure(text="PLAY")
        self.preview_progress.set(self.preview_position / self.preview_duration if self.preview_duration else 0)
        self.preview_timer_label.configure(text=f"{self.format_duration(self.preview_position)} / {self.format_duration(self.preview_duration)}")
        self.preview_status_label.configure(text="Stopped")
        self.status_label.configure(text="Preview stopped")

    def update_preview_timer(self):
        if not self.preview_playing:
            return
        self.preview_position = time.time() - self.preview_started_at
        if self.preview_position >= self.preview_duration:
            self.preview_position = self.preview_duration
            self.preview_playing = False
            sd.stop()
            self.preview_button.configure(text="PLAY")
            self.preview_progress.set(1)
            self.preview_timer_label.configure(text=f"{self.format_duration(self.preview_position)} / {self.format_duration(self.preview_duration)}")
            self.preview_status_label.configure(text="Finished")
            self.status_label.configure(text="Preview finished")
            self.preview_timer_job = None
            return
        self.preview_progress.set(self.preview_position / self.preview_duration if self.preview_duration else 0)
        self.preview_timer_label.configure(text=f"{self.format_duration(self.preview_position)} / {self.format_duration(self.preview_duration)}")
        self.preview_timer_job = self.after(50, self.update_preview_timer)

    def save_as_wav(self):
        if self.preview_data is None or self.preview_samplerate is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"recording_{timestamp}.wav"
        file_path = filedialog.asksaveasfilename(title="Save Recording", defaultextension=".wav", initialfile=default_name, filetypes=[("WAV Audio", "*.wav")])
        if not file_path:
            return
        try:
            sf.write(file_path, self.preview_data, self.preview_samplerate)
            self.status_label.configure(text=f"Saved: {os.path.basename(file_path)}")
        except Exception as error:
            self.status_label.configure(text=f"Save error: {error}")

    def apply_source(self):
        if self.preview_data is None or self.preview_samplerate is None:
            self.status_label.configure(text="Record something first")
            return
        if self.apply_on_source_path is None:
            self.status_label.configure(text="No source target is configured")
            return
        try:
            self.apply_on_source_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(self.apply_on_source_path, self.preview_data, self.preview_samplerate)
            self.status_label.configure(text=f"Applied: {self.apply_on_source_path.name}")
            self.after(220, self.destroy)
        except Exception as error:
            self.status_label.configure(text=f"Apply error: {error}")

    def update_record_timer(self):
        if not self.recording_session:
            return
        elapsed = self.recorded_duration + (time.time() - self.record_segment_start if self.recording else 0)
        self.timer_label.configure(text=self.format_duration(elapsed))
        self.record_timer_job = self.after(100, self.update_record_timer)

    def update_volume_meter(self):
        if not self.recording_session:
            self.volume_bar.set(0)
            self.volume_label.configure(text="0%")
            return
        level = self.volume_level if self.recording else 0.0
        self.volume_bar.set(level)
        self.volume_label.configure(text=f"{int(level * 100)}%")
        self.volume_timer_job = self.after(50, self.update_volume_meter)

    def set_recording_indicator(self, text, color):
        self.indicator_label.configure(text=text, text_color=color)

    def cancel_record_timers(self):
        if self.record_timer_job is not None:
            self.after_cancel(self.record_timer_job)
            self.record_timer_job = None
        if self.volume_timer_job is not None:
            self.after_cancel(self.volume_timer_job)
            self.volume_timer_job = None
        self.volume_bar.set(0)
        self.volume_label.configure(text="0%")

    def cancel_preview_timer(self):
        if self.preview_timer_job is not None:
            self.after_cancel(self.preview_timer_job)
            self.preview_timer_job = None

    def format_duration(self, seconds):
        seconds = max(0, int(seconds))
        minutes = seconds // 60
        seconds %= 60
        return f"{minutes:02d}:{seconds:02d}"

    def on_close(self):
        if self.recording_session:
            self.stop_recording()
        self.cancel_preview_timer()
        sd.stop()
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except OSError:
                pass
        self.destroy()

if __name__ == "__main__":
    apply_on_source_path = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--apply-on-source":
        apply_on_source_path = sys.argv[2]
    app = MicRecorderApp(apply_on_source_path=apply_on_source_path)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()