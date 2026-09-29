import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.vid2wav import convert_media_to_wav

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_VC_ROOT = BASE_DIR / "seed-vc"
INFERENCE_SCRIPT = SEED_VC_ROOT / "inference.py"
BIGVGAN_FILE = SEED_VC_ROOT / "modules" / "bigvgan" / "bigvgan.py"
DIFFUSION_STEP_OPTIONS = {"Low": 25, "Recommended": 50, "High": 75, "Extreme": 100}
MIN_SEMITONE = -72
MAX_SEMITONE = 72
VOCAL_LEAD_DB = 2.0
MAX_VOCAL_DB = -12.0
PEAK_CEILING_DB = -1.0
MIX_TARGET_DB = -18.0

@dataclass(frozen=True)
class SeedVCSettings:
    steps: int = 50
    cfg: float = 0.80
    f0: bool = False
    auto_f0: bool = True
    pitch: int = 0

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

def match_audio_volume(reference_path, target_path, log=print):
    if not reference_path or not target_path:
        return target_path
    reference_path = Path(reference_path)
    target_path = Path(target_path)
    if not reference_path.exists() or not target_path.exists():
        return target_path
    reference, reference_rate = sf.read(reference_path, dtype="float32", always_2d=True)
    target, target_rate = sf.read(target_path, dtype="float32", always_2d=True)
    if reference_rate != target_rate or reference.size == 0 or target.size == 0:
        return target_path
    length = min(len(reference), len(target))
    reference_mono = reference[:length].mean(axis=1)
    target_mono = target[:length].mean(axis=1)
    reference_rms = float(np.sqrt(np.mean(reference_mono ** 2)))
    target_rms = float(np.sqrt(np.mean(target_mono ** 2)))
    if reference_rms <= 0 or target_rms <= 0:
        return target_path
    gain_db = 20 * np.log10(reference_rms / target_rms)
    gain = 10 ** (gain_db / 20.0)
    adjusted = 0.98 * np.tanh(target * gain / 0.98)
    sf.write(target_path, adjusted, target_rate, subtype="PCM_16")
    log(f"Matched converted voice volume to separated vocal: {gain_db:+.2f} dB")
    return target_path

def measure_active_db(audio, frame=4096, range_db=25.0):
    mono = audio.astype(np.float64).mean(axis=1)
    count = len(mono) // frame
    if count == 0:
        return None
    power = (mono[:count * frame].reshape(count, frame) ** 2).mean(axis=1)
    if not np.isfinite(power).all() or power.max() <= 0:
        return None
    active = power >= np.percentile(power, 95) * 10 ** (-range_db / 10)
    return float(10 * np.log10(power[active].mean()))

def normalize_mix_volume(path, log=print):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mix_db = measure_active_db(audio)
    if mix_db is None:
        return path
    peak_db = 20 * np.log10(max(float(np.abs(audio).max()), 1e-9))
    gain_db = min(MIX_TARGET_DB - mix_db, PEAK_CEILING_DB - peak_db)
    sf.write(path, audio * 10 ** (gain_db / 20.0), rate, subtype="PCM_16")
    log(f"Final mix level: {mix_db:+.2f} dBFS, peak {peak_db:+.2f} dBFS; applying {gain_db:+.2f} dB to target {mix_db + gain_db:+.2f} dBFS.")
    return path

def read_active_db(path):
    if not path or not Path(path).exists():
        return None
    audio, _ = sf.read(path, dtype="float32", always_2d=True)
    return measure_active_db(audio)

def get_mix_vocal_gain_db(instrumental_path, converted_path, target_offset_db=0.0, log=print, reference_path=None):
    instrumental_db = read_active_db(instrumental_path)
    reference_db = read_active_db(reference_path)
    converted_db = read_active_db(converted_path)
    if instrumental_db is None or converted_db is None:
        log("Mix balance unchanged: instrumental or converted voice is silent or missing.")
        return 0.0
    anchor_db = (converted_db if reference_db is None else reference_db) + target_offset_db
    target_vocal_db = max(anchor_db, instrumental_db + VOCAL_LEAD_DB)
    gain_db = target_vocal_db - converted_db
    reference_text = "matched" if reference_db is None else f"{reference_db:+.2f} dBFS"
    log(f"Vocal level: instrumental {instrumental_db:+.2f} dBFS, separated vocal {reference_text}, converted voice {converted_db:+.2f} dBFS; applying {gain_db:+.2f} dB to target {target_vocal_db:+.2f} dBFS.")
    return float(gain_db)

def prepare_source(source_path, inputs_dir, ffmpeg_path, log=print):
    seed_source_wav = Path(inputs_dir) / "seed_source.wav"
    convert_media_to_wav(source_path, seed_source_wav, ffmpeg_path=ffmpeg_path, channels=1, sample_rate=44100, label="source vocal")
    log(f"Source vocal audio ready: {seed_source_wav}")
    return seed_source_wav

def get_config(steps_choice="Recommended", follow_pitch="Target Voice Pitch", mode="Vocalize", semitone=0):
    steps = DIFFUSION_STEP_OPTIONS.get(steps_choice, 50)
    target_pitch = follow_pitch == "Target Voice Pitch"
    vocalize = mode == "Vocalize"
    auto_f0 = target_pitch
    pitch = max(MIN_SEMITONE, min(MAX_SEMITONE, int(semitone)))
    return SeedVCSettings(steps=steps, f0=vocalize, auto_f0=auto_f0, pitch=pitch)

def command(source_wav, target_wav, output_dir, cfg):
    return [sys.executable, str(INFERENCE_SCRIPT), "--source", str(source_wav), "--target", str(target_wav), "--output", str(output_dir), "--diffusion-steps", str(cfg.steps), "--length-adjust", "1.0", "--inference-cfg-rate", str(cfg.cfg), "--f0-condition", str(cfg.f0), "--auto-f0-adjust", str(cfg.auto_f0), "--semi-tone-shift", str(cfg.pitch), "--fp16", "True"]

def run(source_wav, target_wav, output_dir, cfg, log=print, output_name=None):
    output_dir = Path(output_dir)
    for item in output_dir.glob("*.wav"):
        try:
            item.unlink()
        except Exception:
            pass
    process = subprocess.Popen(command(source_wav, target_wav, output_dir, cfg), cwd=str(SEED_VC_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=os.environ.copy())
    for line in iter(process.stdout.readline, ""):
        if line:
            log(line.rstrip())
    process.stdout.close()
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Seed-VC exited with code {code}.")
    outputs = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not outputs:
        raise RuntimeError("Seed-VC produced no WAV output.")
    converted_path = outputs[0]
    if output_name:
        converted_path = converted_path.rename(output_dir / output_name)
    log(f"Converted vocal: {converted_path}")
    return converted_path