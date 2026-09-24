import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import numpy as np
import sounddevice as sd
import soundfile as sf
from utils.centwin import center_window
from utils.ctk_theme import configure_ctk_theme

configure_ctk_theme()

class MicRecorderApp(ctk.CTk):
    def __init__(self, apply_on_source_path=None):
        super().__init__()
        self.apply_on_source_path = Path(apply_on_source_path) if apply_on_source_path else None
        self.title("Microphone Recorder")
        self.geometry("920x900")
        center_window(self, width=920, height=900)
        self.minsize(820, 840)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
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
        self.preview_play_origin = 0.0
        self.preview_position = 0.0
        self.preview_duration = 0.0
        self.preview_timer_job = None
        self.trim_start = 0.0
        self.trim_end = 0.0
        self.drag_target = None
        self.devices = {}
        self.waveform_height = 240
        self.build_ui()
        self.refresh_devices()

    def build_ui(self):
        device_frame = ctk.CTkFrame(self)
        device_frame.grid(row=1, column=0, padx=38, pady=(24, 16), sticky="ew")
        device_frame.grid_columnconfigure(1, weight=1)
        device_label = ctk.CTkLabel(device_frame, text="MICROPHONE", font=ctk.CTkFont(size=13, weight="bold"))
        device_label.grid(row=0, column=0, padx=(20, 15), pady=18)
        self.device_menu = ctk.CTkOptionMenu(device_frame, values=["No active microphone found"], height=40)
        self.device_menu.grid(row=0, column=1, padx=10, pady=18, sticky="ew")
        self.refresh_button = ctk.CTkButton(device_frame, text="REFRESH", width=110, height=40, command=self.refresh_devices)
        self.refresh_button.grid(row=0, column=2, padx=(10, 20), pady=18)
        content = ctk.CTkFrame(self, fg_color="transparent", border_width=0, corner_radius=0)
        content.grid(row=2, column=0, padx=38, pady=(0, 16), sticky="ew")
        content.grid_rowconfigure(0, minsize=270)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1)
        record_frame = ctk.CTkFrame(content)
        record_frame.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        record_frame.grid_columnconfigure(0, weight=1)
        record_header = ctk.CTkFrame(record_frame, fg_color="transparent", border_width=0, corner_radius=0)
        record_header.grid(row=0, column=0, padx=22, pady=(18, 6), sticky="ew")
        record_header.grid_columnconfigure(0, weight=1)
        record_title = ctk.CTkLabel(record_header, text="RECORDING", font=ctk.CTkFont(size=18, weight="bold"))
        record_title.grid(row=0, column=0, sticky="w")
        self.indicator_label = ctk.CTkLabel(record_header, text="● READY", font=ctk.CTkFont(size=13, weight="bold"), text_color="gray")
        self.indicator_label.grid(row=0, column=1, sticky="e")
        self.status_label = ctk.CTkLabel(record_frame, text="Ready to record", font=ctk.CTkFont(size=14))
        self.status_label.grid(row=1, column=0, padx=20, pady=(5, 0))
        self.timer_label = ctk.CTkLabel(record_frame, text="00:00", font=ctk.CTkFont(size=52, weight="bold"))
        self.timer_label.grid(row=2, column=0, padx=20, pady=(0, 8))
        record_controls = ctk.CTkFrame(record_frame, fg_color="transparent", border_width=0, corner_radius=0)
        record_controls.grid(row=3, column=0, padx=20, pady=(0, 8))
        self.record_button = ctk.CTkButton(record_controls, text="RECORD", width=150, height=44, command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=6)
        self.stop_button = ctk.CTkButton(record_controls, text="STOP", width=150, height=44, command=self.stop_recording, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=6)
        volume_frame = ctk.CTkFrame(content)
        volume_frame.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        volume_title = ctk.CTkLabel(volume_frame, text="INPUT VOLUME", font=ctk.CTkFont(size=15, weight="bold"))
        volume_title.pack(pady=(20, 8))
        self.volume_bar = ctk.CTkProgressBar(volume_frame, orientation="vertical", width=34, height=150)
        self.volume_bar.pack(pady=6)
        self.volume_bar.set(0)
        self.volume_label = ctk.CTkLabel(volume_frame, text="0%", font=ctk.CTkFont(size=14, weight="bold"))
        self.volume_label.pack(pady=(6, 2))
        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(row=3, column=0, padx=38, pady=(0, 25), sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(2, minsize=self.waveform_height, weight=0)
        preview_header = ctk.CTkFrame(preview_frame, fg_color="transparent", border_width=0, corner_radius=0)
        preview_header.grid(row=0, column=0, padx=22, pady=(18, 5), sticky="ew")
        preview_header.grid_columnconfigure(0, weight=1)
        preview_title = ctk.CTkLabel(preview_header, text="AUDIO PREVIEW", font=ctk.CTkFont(size=18, weight="bold"))
        preview_title.grid(row=0, column=0, sticky="w")
        self.preview_status_label = ctk.CTkLabel(preview_header, text="No recording", font=ctk.CTkFont(size=13))
        self.preview_status_label.grid(row=0, column=1, sticky="e")
        self.waveform_canvas = tk.Canvas(preview_frame, height=self.waveform_height, background="#1c1c1c", highlightthickness=0, cursor="crosshair")
        self.waveform_canvas.grid(row=2, column=0, padx=22, pady=(0, 10), sticky="ew")
        self.waveform_canvas.bind("<Configure>", self.on_waveform_configure)
        self.waveform_canvas.bind("<Button-1>", self.on_waveform_press)
        self.waveform_canvas.bind("<B1-Motion>", self.on_waveform_drag)
        self.waveform_canvas.bind("<ButtonRelease-1>", self.on_waveform_release)
        self.trim_label = ctk.CTkLabel(preview_frame, text="START 00:00    END 00:00", font=ctk.CTkFont(size=13, weight="bold"))
        self.trim_label.grid(row=3, column=0, padx=20, pady=(2, 3))
        self.preview_timer_label = ctk.CTkLabel(preview_frame, text="00:00 / 00:00", font=ctk.CTkFont(size=18, weight="bold"))
        self.preview_timer_label.grid(row=5, column=0, padx=20, pady=4)
        preview_controls = ctk.CTkFrame(preview_frame, fg_color="transparent", border_width=0, corner_radius=0)
        preview_controls.grid(row=6, column=0, padx=20, pady=(7, 16))
        self.preview_button = ctk.CTkButton(preview_controls, text="►", width=54, height=42, font=ctk.CTkFont(family="Segoe UI Symbol", size=17), command=self.toggle_preview, state="disabled")
        self.preview_button.grid(row=0, column=0, padx=5)
        self.restart_button = ctk.CTkButton(preview_controls, text="⏮", width=54, height=42, font=ctk.CTkFont(family="Segoe UI Symbol", size=17), command=self.restart_preview, state="disabled")
        self.restart_button.grid(row=0, column=1, padx=5)
        self.save_button = ctk.CTkButton(preview_controls, text="SAVE AS WAV", width=150, height=42, command=self.save_as_wav, state="disabled")
        self.save_button.grid(row=0, column=2, padx=5)
        self.apply_button = ctk.CTkButton(preview_controls, text="APPLY ON SOURCE", width=180, height=42, command=self.apply_source, state="normal" if self.apply_on_source_path else "disabled")
        self.apply_button.grid(row=0, column=3, padx=5)

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
            self.preview_duration = 0.0
            self.preview_position = 0.0
            self.trim_start = 0.0
            self.trim_end = 0.0
            self.clear_waveform()
            self.recorded_duration = 0.0
            self.record_segment_start = time.monotonic()
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
            self.restart_button.configure(state="disabled")
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
        self.recorded_duration += time.monotonic() - self.record_segment_start
        self.recording = False
        self.paused = True
        self.volume_level = 0.0
        self.record_button.configure(text="PLAY")
        self.set_recording_indicator("● PAUSED", "orange")
        self.status_label.configure(text="Recording paused")

    def resume_recording(self):
        if not self.recording_session or not self.paused:
            return
        self.record_segment_start = time.monotonic()
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
            self.recorded_duration += time.monotonic() - self.record_segment_start
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
        self.trim_start = 0.0
        self.trim_end = self.preview_duration
        try:
            if self.temp_path and os.path.exists(self.temp_path):
                os.remove(self.temp_path)
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            self.temp_path = temp_file.name
            temp_file.close()
            sf.write(self.temp_path, self.preview_data, self.preview_samplerate)
            self.preview_timer_label.configure(text=f"00:00 / {self.format_duration(self.preview_duration)}")
            self.preview_button.configure(state="normal", text="►")
            self.restart_button.configure(state="normal")
            self.save_button.configure(state="normal")
            self.preview_status_label.configure(text="Ready to preview")
            self.set_recording_indicator("● RECORDED", "gray")
            self.status_label.configure(text="Recording ready")
            self.update_trim_label()
            self.draw_waveform()
        except Exception as error:
            self.set_recording_indicator("● READY", "gray")
            self.status_label.configure(text=f"Temporary save error: {error}")

    def toggle_preview(self):
        if self.preview_data is None:
            return
        if self.preview_playing:
            self.pause_preview()
        else:
            self.play_preview()

    def play_preview(self):
        if self.preview_data is None or self.preview_samplerate is None or self.preview_duration <= 0:
            return
        if self.preview_position >= self.trim_end or self.preview_position < self.trim_start:
            self.preview_position = self.trim_start
        start_sample = int(self.preview_position * self.preview_samplerate)
        end_sample = max(start_sample + 1, int(self.trim_end * self.preview_samplerate))
        try:
            sd.stop()
            sd.play(self.preview_data[start_sample:end_sample], self.preview_samplerate)
            self.preview_play_origin = self.preview_position
            self.preview_started_at = time.monotonic()
            self.preview_playing = True
            self.preview_button.configure(text="❚❚")
            self.preview_status_label.configure(text="Playing")
            self.status_label.configure(text="Playing preview")
            self.cancel_preview_timer()
            self.update_preview_timer()
        except Exception as error:
            self.preview_playing = False
            self.preview_button.configure(text="►")
            self.preview_status_label.configure(text="Ready to preview")
            self.status_label.configure(text=f"Playback error: {error}")

    def pause_preview(self):
        if not self.preview_playing:
            return
        self.preview_position = min(self.preview_play_origin + time.monotonic() - self.preview_started_at, self.trim_end)
        self.preview_playing = False
        sd.stop()
        self.cancel_preview_timer()
        self.preview_button.configure(text="►")
        self.preview_status_label.configure(text="Paused")
        self.status_label.configure(text="Preview paused")
        self.update_playhead()
        self.update_preview_labels()

    def restart_preview(self):
        if self.preview_data is None:
            return
        self.preview_playing = False
        sd.stop()
        self.cancel_preview_timer()
        self.preview_position = self.trim_start
        self.preview_button.configure(text="►", state="normal")
        self.preview_status_label.configure(text="Ready to preview")
        self.status_label.configure(text="Preview reset to start")
        self.update_playhead()
        self.update_preview_labels()

    def update_preview_timer(self):
        if not self.preview_playing:
            return
        self.preview_position = min(self.preview_play_origin + time.monotonic() - self.preview_started_at, self.trim_end)
        self.update_playhead()
        self.update_preview_labels()
        if self.preview_position >= self.trim_end:
            self.preview_playing = False
            sd.stop()
            self.cancel_preview_timer()
            self.preview_button.configure(text="►")
            self.preview_status_label.configure(text="Finished")
            self.status_label.configure(text="Preview finished")
            return
        self.preview_timer_job = self.after(50, self.update_preview_timer)

    def on_waveform_configure(self, event=None):
        if self.preview_data is not None:
            self.draw_waveform()

    def waveform_time_from_x(self, x):
        width = max(1, self.waveform_canvas.winfo_width())
        return min(self.preview_duration, max(0.0, x / width * self.preview_duration))

    def waveform_x_from_time(self, seconds):
        width = max(1, self.waveform_canvas.winfo_width())
        return seconds / self.preview_duration * width if self.preview_duration else 0.0

    def on_waveform_press(self, event):
        if self.preview_data is None:
            return
        x = event.x
        start_x = self.waveform_x_from_time(self.trim_start)
        end_x = self.waveform_x_from_time(self.trim_end)
        playhead_x = self.waveform_x_from_time(self.preview_position)
        if abs(x - start_x) <= 10:
            self.stop_preview_for_edit()
            self.drag_target = "start"
            return
        if abs(x - end_x) <= 10:
            self.stop_preview_for_edit()
            self.drag_target = "end"
            return
        if abs(x - playhead_x) <= 8:
            self.stop_preview_for_edit()
            self.drag_target = "playhead"
            return
        self.drag_target = None
        self.seek_preview(self.waveform_time_from_x(x))

    def on_waveform_drag(self, event):
        if self.preview_data is None or self.drag_target is None:
            return
        position = self.waveform_time_from_x(event.x)
        gap = max(0.01, self.preview_duration / 1000)
        if self.drag_target == "start":
            self.trim_start = min(position, self.trim_end - gap)
            self.preview_position = max(self.preview_position, self.trim_start)
        elif self.drag_target == "end":
            self.trim_end = max(position, self.trim_start + gap)
            self.preview_position = min(self.preview_position, self.trim_end)
        else:
            self.preview_position = min(self.trim_end, max(self.trim_start, position))
        self.update_trim_label()
        self.update_preview_labels()
        self.draw_waveform()

    def on_waveform_release(self, event=None):
        self.drag_target = None

    def stop_preview_for_edit(self):
        self.preview_playing = False
        sd.stop()
        self.cancel_preview_timer()
        self.preview_button.configure(text="►")
        self.update_preview_labels()

    def seek_preview(self, position):
        if self.preview_data is None:
            return
        self.preview_playing = False
        sd.stop()
        self.cancel_preview_timer()
        self.preview_position = min(self.trim_end, max(self.trim_start, position))
        self.preview_button.configure(text="►")
        self.preview_status_label.configure(text="Ready to preview")
        self.update_playhead()
        self.update_preview_labels()

    def draw_waveform(self):
        self.waveform_canvas.delete("all")
        if self.preview_data is None or self.preview_duration <= 0:
            return
        width = max(1, self.waveform_canvas.winfo_width())
        height = self.waveform_height
        center = height / 2
        samples = self.preview_data[:, 0] if self.preview_data.ndim > 1 else self.preview_data
        count = max(100, min(1800, width))
        chunk_size = max(1, len(samples) // count)
        amplitudes = []
        for index in range(count):
            chunk = samples[index * chunk_size:(index + 1) * chunk_size]
            if len(chunk) == 0:
                break
            amplitudes.append(float(np.max(np.abs(chunk))))
        if not amplitudes:
            return
        start_x = self.waveform_x_from_time(self.trim_start)
        end_x = self.waveform_x_from_time(self.trim_end)
        self.waveform_canvas.create_rectangle(0, 0, start_x, height, fill="#141414", outline="")
        self.waveform_canvas.create_rectangle(end_x, 0, width, height, fill="#141414", outline="")
        step = width / len(amplitudes)
        for index, amplitude in enumerate(amplitudes):
            x = index * step + step / 2
            sample_time = (index + 0.5) / len(amplitudes) * self.preview_duration
            active = self.trim_start <= sample_time <= self.trim_end
            peak = max(2.0, amplitude * (height * 0.42))
            self.waveform_canvas.create_line(x, center - peak, x, center + peak, fill="#62a0ea" if active else "#4a4a4a", width=2)
        self.waveform_canvas.create_line(0, center, width, center, fill="#303030")
        self.waveform_canvas.create_line(start_x, 12, start_x, height - 12, fill="#9ad1ff", width=4)
        self.waveform_canvas.create_line(end_x, 12, end_x, height - 12, fill="#9ad1ff", width=4)
        self.waveform_canvas.create_polygon(start_x, 4, start_x - 8, 16, start_x + 8, 16, fill="#9ad1ff", outline="")
        self.waveform_canvas.create_polygon(end_x, 4, end_x - 8, 16, end_x + 8, 16, fill="#9ad1ff", outline="")
        self.update_playhead()

    def update_playhead(self):
        if self.preview_data is None or self.preview_duration <= 0:
            return
        x = self.waveform_x_from_time(self.preview_position)
        self.waveform_canvas.delete("playhead")
        self.waveform_canvas.create_line(x, 0, x, self.waveform_height, fill="#ffffff", width=3, tags="playhead")

    def update_preview_labels(self):
        current = max(0.0, self.preview_position - self.trim_start)
        total = max(0.0, self.trim_end - self.trim_start)
        self.preview_timer_label.configure(text=f"{self.format_duration(current)} / {self.format_duration(total)}")

    def update_trim_label(self):
        self.trim_label.configure(text=f"START {self.format_duration(self.trim_start)}    END {self.format_duration(self.trim_end)}")

    def save_as_wav(self):
        if self.preview_data is None or self.preview_samplerate is None:
            return
        start_sample = int(self.trim_start * self.preview_samplerate)
        end_sample = int(self.trim_end * self.preview_samplerate)
        trimmed_data = self.preview_data[start_sample:end_sample]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"recording_{timestamp}.wav"
        file_path = filedialog.asksaveasfilename(title="Save Recording", defaultextension=".wav", initialfile=default_name, filetypes=[("WAV Audio", "*.wav")])
        if not file_path:
            return
        try:
            sf.write(file_path, trimmed_data, self.preview_samplerate)
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
            start_sample = int(self.trim_start * self.preview_samplerate)
            end_sample = int(self.trim_end * self.preview_samplerate)
            trimmed_data = self.preview_data[start_sample:end_sample]
            self.apply_on_source_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(self.apply_on_source_path, trimmed_data, self.preview_samplerate)
            self.status_label.configure(text=f"Applied: {self.apply_on_source_path.name}")
            self.after(220, self.destroy)
        except Exception as error:
            self.status_label.configure(text=f"Apply error: {error}")

    def update_record_timer(self):
        if not self.recording_session:
            return
        elapsed = self.recorded_duration + (time.monotonic() - self.record_segment_start if self.recording else 0)
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

    def clear_waveform(self):
        self.waveform_canvas.delete("all")
        self.trim_label.configure(text="START 00:00    END 00:00")
        self.preview_timer_label.configure(text="00:00 / 00:00")

    def format_duration(self, seconds):
        seconds = max(0, int(seconds))
        minutes = seconds // 60
        seconds %= 60
        return f"{minutes:02d}:{seconds:02d}"

    def on_close(self):
        if self.recording_session:
            self.stop_recording()
        self.preview_playing = False
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