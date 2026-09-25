import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from utils.vid2wav import convert_media_to_wav

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_VC_ROOT = BASE_DIR / "seed-vc"
INFERENCE_SCRIPT = SEED_VC_ROOT / "inference.py"
BIGVGAN_FILE = SEED_VC_ROOT / "modules" / "bigvgan" / "bigvgan.py"
DIFFUSION_STEP_OPTIONS = {"Low": 25, "Recommended": 50, "High": 75, "Extreme": 100}
MIN_SEMITONE = -72
MAX_SEMITONE = 72


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

def run(source_wav, target_wav, output_dir, cfg, log=print):
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
    log(f"Converted vocal: {converted_path}")
    return converted_path