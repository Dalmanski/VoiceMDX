import os
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_VC_ROOT = BASE_DIR / "seed-vc"
INFERENCE_SCRIPT = SEED_VC_ROOT / "inference.py"
BIGVGAN_FILE = SEED_VC_ROOT / "modules" / "bigvgan" / "bigvgan.py"
DIFFUSION_STEP_OPTIONS = {"Low": 25, "Recommended": 50, "High": 75, "Extreme": 100}


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


def prepare_source(app, source_path=None):
    app.seed_source_wav = app.inputs_dir / "seed_source.wav"
    input_source = source_path or app.uvr_vocal_path
    app.extract_audio(input_source, app.seed_source_wav, "source vocal", 1)
    app.seed_source_wav = app.normalize_audio(app.seed_source_wav, "seed_source")


def get_config(app):
    steps = DIFFUSION_STEP_OPTIONS.get(app.steps_choice_var.get(), 50)
    target_pitch = app.follow_pitch_var.get() == "Target Voice Pitch"
    vocalize = app.mode_var.get() == "Vocalize"
    return {"steps": steps, "cfg": 0.80, "f0": vocalize, "auto_f0": target_pitch if vocalize else not target_pitch, "pitch": 0}


def command(app):
    cfg = get_config(app)
    return [sys.executable, str(INFERENCE_SCRIPT), "--source", str(app.seed_source_wav), "--target", str(app.target_wav), "--output", str(app.seed_output_dir), "--diffusion-steps", str(cfg["steps"]), "--length-adjust", "1.0", "--inference-cfg-rate", str(cfg["cfg"]), "--f0-condition", str(cfg["f0"]), "--auto-f0-adjust", str(cfg["auto_f0"]), "--semi-tone-shift", "0", "--fp16", "True"]


def run(app):
    for item in app.seed_output_dir.glob("*.wav"):
        try:
            item.unlink()
        except Exception:
            pass
    app.process = subprocess.Popen(command(app), cwd=str(SEED_VC_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=os.environ.copy())
    for line in iter(app.process.stdout.readline, ""):
        if line:
            app.log(line.rstrip())
    app.process.stdout.close()
    code = app.process.wait()
    app.process = None
    if code != 0:
        raise RuntimeError(f"Seed-VC exited with code {code}.")
    outputs = sorted(app.seed_output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not outputs:
        raise RuntimeError("Seed-VC produced no WAV output.")
    app.converted_vocal_path = outputs[0]
    app.log(f"Converted vocal: {app._display_path(app.converted_vocal_path)}")