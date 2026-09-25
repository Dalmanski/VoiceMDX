import atexit
import math
import shutil
import subprocess
import tempfile
from importlib import import_module
from pathlib import Path
import numpy as np
import soundfile as sf
from utils.vid2wav import convert_media_to_wav

run_target_uvr = import_module("utils.uvr-mdx").run_target_uvr

MAX_TARGET_SECONDS = 25.0
MIN_TARGET_SECONDS = 1.0
TRIM_SILENCE_DBFS = -60.0
TRIM_WINDOW_SECONDS = 0.05
TRIM_PADDING_SECONDS = 0.05
TEMP_ROOT = Path(tempfile.gettempdir()) / "voicemdx_temp"
_TEMP_DIRS = set()

def _cleanup():
    for path in tuple(_TEMP_DIRS):
        shutil.rmtree(path, ignore_errors=True)
        _TEMP_DIRS.discard(path)

atexit.register(_cleanup)

def _temp_dir(prefix):
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=str(TEMP_ROOT)))
    _TEMP_DIRS.add(str(path))
    return path

def _ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg was not found on PATH.")
    return path

def _run(command, timeout=180):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg timed out after {timeout} seconds.") from exc
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or "FFmpeg failed.")
    return result.stdout

def _trim_silence_edges(input_path, output_path):
    with sf.SoundFile(input_path) as audio:
        sample_rate = audio.samplerate
        channels = audio.channels
        data = audio.read(dtype="float32", always_2d=True)
    if data.size == 0:
        raise RuntimeError("UVR vocal contains no samples.")
    mono = data.mean(axis=1)
    threshold = 10.0 ** (TRIM_SILENCE_DBFS / 20.0)
    window = max(1, int(sample_rate * TRIM_WINDOW_SECONDS))
    padding = int(sample_rate * TRIM_PADDING_SECONDS)
    frame_count = max(1, math.ceil(len(mono) / window))
    rms_values = np.array([float(np.sqrt(np.mean(mono[index * window:min((index + 1) * window, len(mono))] ** 2))) for index in range(frame_count)])
    active = np.flatnonzero(rms_values > threshold)
    if active.size == 0:
        sf.write(output_path, data, sample_rate, subtype="PCM_16")
        return False
    start = max(0, int(active[0]) * window - padding)
    end = min(len(data), min(len(mono), int(active[-1] + 1) * window) + padding)
    trimmed = data[start:end]
    if len(trimmed) < int(sample_rate * MIN_TARGET_SECONDS):
        sf.write(output_path, data, sample_rate, subtype="PCM_16")
        return False
    sf.write(output_path, trimmed, sample_rate, subtype="PCM_16")
    return start > 0 or end < len(data)

def _prepare_seed_vc(input_path, output_path):
    trimmed_path = output_path.with_name("trimmed_vocal.wav")
    _trim_silence_edges(input_path, trimmed_path)
    command = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(trimmed_path), "-map", "0:a:0", "-vn", "-ac", "1", "-t", str(MAX_TARGET_SECONDS), "-af", "dynaudnorm=f=150:g=15", "-c:a", "pcm_s16le", "-y", str(output_path)]
    _run(command)
    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError(f"Seed-VC preparation produced an empty file: {output_path}")
    return output_path

def _analyze(path):
    with sf.SoundFile(path) as audio:
        sample_rate = audio.samplerate
        channels = audio.channels
        frames = audio.frames
        data = audio.read(dtype="float32", always_2d=True)
    if data.size == 0:
        raise RuntimeError("Audio contains no samples.")
    mono = data.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak_dbfs = 20.0 * math.log10(peak) if peak else float("-inf")
    rms_dbfs = 20.0 * math.log10(rms) if rms else float("-inf")
    silence_ratio = float(np.mean(np.abs(mono) < 0.008))
    return {"sample_rate": sample_rate, "channels": channels, "duration": frames / sample_rate if sample_rate else 0.0, "peak_dbfs": peak_dbfs, "rms_dbfs": rms_dbfs, "silence_ratio": silence_ratio}

def prepare_seed_vc_target(path, max_seconds=MAX_TARGET_SECONDS, min_seconds=MIN_TARGET_SECONDS, log=print):
    source = Path(path).resolve()
    if not source.is_file():
        raise RuntimeError(f"Target file was not found: {source}")
    ffmpeg_path = _ffmpeg()
    work_dir = _temp_dir("seedvc_target_")
    inputs_dir = work_dir / "inputs"
    target_vocal_dir = work_dir / "target_vocal"
    preview_path = work_dir / "original_preview.wav"
    prepared_path = work_dir / "seedvc_ready.wav"
    inputs_dir.mkdir()
    target_vocal_dir.mkdir()
    convert_media_to_wav(source, preview_path, ffmpeg_path=ffmpeg_path, channels=2, sample_rate=44100, label="original_preview")
    target_vocal_path = Path(run_target_uvr(source, ffmpeg_path, inputs_dir, target_vocal_dir, log=log))
    if not target_vocal_path.exists():
        raise RuntimeError(f"UVR target vocal was not found: {target_vocal_path}")
    source_info = _analyze(target_vocal_path)
    issues = []
    if source_info["duration"] < min_seconds:
        issues.append("shorter than 1 second")
    if source_info["duration"] > max_seconds:
        issues.append("longer than 25 seconds")
    if source_info["channels"] != 1:
        issues.append(f"{source_info['channels']} channels")
    if source_info["peak_dbfs"] <= -45.0:
        issues.append("too quiet")
    if source_info["silence_ratio"] >= 0.85:
        issues.append("mostly silent")
    _prepare_seed_vc(target_vocal_path, prepared_path)
    prepared_info = _analyze(prepared_path)
    trimmed_silence = prepared_info["duration"] < min(source_info["duration"], max_seconds) - 0.05
    changes = list(issues)
    if trimmed_silence:
        changes.append("trimmed leading/trailing silence")
    converted = bool(changes)
    message = "UVR vocal extracted and automatically prepared: " + ", ".join(changes) + "." if changes else "UVR vocal passed the technical checks and was packaged as a Seed-VC-ready WAV."
    if prepared_info["duration"] < min_seconds:
        raise RuntimeError(f"Prepared target vocal is shorter than {min_seconds:.0f} second.")
    if prepared_info["duration"] > max_seconds:
        raise RuntimeError("Prepared target vocal is longer than 25 seconds.")
    if prepared_info["channels"] != 1:
        raise RuntimeError("Prepared target vocal is not mono.")
    return {"valid": True, "converted": converted, "trimmed_silence": trimmed_silence, "message": message, "original_path": str(source), "original_preview_path": str(preview_path), "vocal_path": str(target_vocal_path), "prepared_path": str(prepared_path), "prepared_bytes": prepared_path.read_bytes(), "duration": prepared_info["duration"], "sample_rate": prepared_info["sample_rate"], "channels": prepared_info["channels"], "peak_dbfs": prepared_info["peak_dbfs"], "rms_dbfs": prepared_info["rms_dbfs"], "silence_ratio": prepared_info["silence_ratio"], "issues": issues}