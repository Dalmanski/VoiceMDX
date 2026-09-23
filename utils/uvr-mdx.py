from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import onnxruntime as ort
    try:
        ort.preload_dlls()
    except Exception:
        pass
    ONNX_RUNTIME_IMPORT_ERROR = None
except Exception as exc:
    ort = None
    ONNX_RUNTIME_IMPORT_ERROR = exc

try:
    from audio_separator.separator import Separator
    AUDIO_SEPARATOR_IMPORT_ERROR = None
except Exception as exc:
    Separator = None
    AUDIO_SEPARATOR_IMPORT_ERROR = exc

BASE_DIR = Path(__file__).resolve().parent.parent
UVR_MODEL_DIR = BASE_DIR / "UVR_MODELS"
UVR_INSTRUMENT_MODEL = UVR_MODEL_DIR / "UVR-MDX-NET-Inst_HQ_4.onnx"
UVR_VOCAL_MODEL = UVR_MODEL_DIR / "UVR-MDX-NET-Voc_FT.onnx"
UVR_BATCH_SIZE = 2

def validate(app):
    if Separator is None:
        raise RuntimeError(f"audio-separator import failed: {type(AUDIO_SEPARATOR_IMPORT_ERROR).__name__}: {AUDIO_SEPARATOR_IMPORT_ERROR}")
    if not UVR_VOCAL_MODEL.exists():
        raise RuntimeError(f"Missing vocal model: {UVR_VOCAL_MODEL}")

def create_separator(app, output_dir, stem_name, batch_size=UVR_BATCH_SIZE):
    if ort is None:
        raise RuntimeError(f"ONNX Runtime import failed: {type(ONNX_RUNTIME_IMPORT_ERROR).__name__}: {ONNX_RUNTIME_IMPORT_ERROR}")
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"UVR CUDAExecutionProvider is unavailable. Available providers: {providers}")
    separator = Separator(output_dir=str(output_dir), model_file_dir=str(UVR_MODEL_DIR), output_format="WAV", sample_rate=44100, use_soundfile=True, use_autocast=False, output_single_stem=stem_name, mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": batch_size, "enable_denoise": False})
    separator.load_model(model_filename=UVR_INSTRUMENT_MODEL.name if stem_name.lower() == "instrumental" else UVR_VOCAL_MODEL.name)
    provider = getattr(separator, "onnx_execution_provider", None)
    if provider and provider != ["CUDAExecutionProvider"]:
        raise RuntimeError(f"UVR selected {provider} instead of CUDAExecutionProvider")
    app.log(f"UVR CUDA: CUDAExecutionProvider | batch={batch_size}")
    return separator

def flatten_paths(value):
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, dict):
        value = value.values()
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(flatten_paths(item))
        return result
    return []

def pick_stem_path(paths, name):
    key = name.lower()
    return next((path for path in paths if path.exists() and key in path.stem.lower()), None)

def has_background(source_path, vocal_path):
    try:
        source = sf.read(source_path, dtype="float32", always_2d=True, stop=44100 * 30)[0]
        vocal = sf.read(vocal_path, dtype="float32", always_2d=True, stop=44100 * 30)[0]
        length = min(len(source), len(vocal))
        source = source[:length].mean(axis=1)
        vocal = vocal[:length].mean(axis=1)
        gain = np.dot(source, vocal) / max(np.dot(vocal, vocal), 1e-12)
        residual = source - vocal * gain
        ratio = np.sqrt(np.mean(residual ** 2) / max(np.mean(source ** 2), 1e-12))
        return ratio > 0.2
    except Exception:
        return True

def separate(app, input_path, model_path, output_dir, stem_name, retry_message, missing_message):
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.glob("*.wav"):
        try:
            item.unlink()
        except Exception:
            pass
    app.log(f"Loading {model_path.name}")
    separator = create_separator(app, output_dir, stem_name)
    try:
        result = separator.separate(str(input_path))
    except Exception as exc:
        if UVR_BATCH_SIZE <= 1 or "memory" not in str(exc).lower():
            raise
        app.log(retry_message)
        app.cleanup_gpu()
        result = create_separator(app, output_dir, stem_name, 1).separate(str(input_path))
    paths = flatten_paths(result)
    discovered = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    paths += [path for path in discovered if path not in paths]
    selected = pick_stem_path(paths, stem_name) or (paths[0] if paths else None)
    if not selected or not selected.exists():
        raise RuntimeError(missing_message)
    return selected

def ensure_source_stems(app):
    if app.inst_path and app.inst_path.exists() and app.uvr_vocal_path and app.uvr_vocal_path.exists():
        return
    if not app.source_path or not app.source_path.exists():
        raise RuntimeError("Source audio or video was not found.")
    if not app.ffmpeg:
        raise RuntimeError(f"FFmpeg was not found: {app.ffmpeg}")
    validate(app)
    app.source_wav = app.inputs_dir / "source.wav"
    app.ext_audio(app.source_path, app.source_wav, "source", 2)
    app.uvr_vocal_path = separate(app, app.source_wav, UVR_VOCAL_MODEL, app.vocal_dir, "Vocals", "UVR batch 2 memory error; retrying with batch 1...", f"{UVR_VOCAL_MODEL.name} did not produce Vocals.wav")
    app.cleanup_gpu()
    app.has_background = has_background(app.source_wav, app.uvr_vocal_path)
    app.log(f"Source background detected: {'yes' if app.has_background else 'no'}")
    if app.has_background:
        if not UVR_INSTRUMENT_MODEL.exists():
            raise RuntimeError(f"Missing instrument model: {UVR_INSTRUMENT_MODEL}")
        app.inst_path = separate(app, app.source_wav, UVR_INSTRUMENT_MODEL, app.inst_dir, "Instrumental", "UVR batch 2 memory error; retrying with batch 1...", f"{UVR_INSTRUMENT_MODEL.name} did not produce Instrumental.wav")
        app.cleanup_gpu()
        app.inst_path = app.norm_audio(app.inst_path, "instrumental")
    else:
        app.inst_path = None
        app.after(0, lambda: app.inst_name.configure(text="Vocal only on source", text_color="yellow"))
    app.uvr_vocal_path = app.norm_audio(app.uvr_vocal_path, "source_vocal")
    if app.has_background:
        app.after(0, lambda: app.inst_name.configure(text=app.inst_path.name, text_color="green"))
        app.after(0, lambda: app.inst_prev.configure(state="normal"))
        app.after(0, lambda: app.inst_dl.configure(state="normal"))
    app.after(0, lambda: app.vocal_name.configure(text=app.uvr_vocal_path.name, text_color="green"))
    app.after(0, lambda: app.vocal_prev.configure(state="normal"))
    app.after(0, lambda: app.vocal_dl.configure(state="normal"))
    if app.inst_path:
        app.log(f"Normalized instrumental: {app._display_path(app.inst_path)}")
    else:
        app.log("Normalized instrumental: not available (no instrument/BG detected)")
    app.log(f"Normalized source vocal: {app._display_path(app.uvr_vocal_path)}")

def run_target_uvr(app):
    if not app.target_path or not app.target_path.exists():
        raise RuntimeError("Target voice file not found.")
    if not app.ffmpeg:
        raise RuntimeError(f"FFmpeg was not found: {app.ffmpeg}")
    validate(app)
    target_input = app.inputs_dir / "target.wav"
    app.ext_audio(app.target_path, target_input, "target", 1)
    selected = separate(app, target_input, UVR_VOCAL_MODEL, app.target_vocal_dir, "Vocals", "UVR batch 2 memory error; retrying target with batch 1...", "Voc_FT did not produce a cleaned target vocal.")
    app.target_vocal_path = app.norm_audio(selected, "target_vocal")
    app.target_wav = app.target_preview_wav = app.target_vocal_path
    app.log(f"Clean target vocal: {app._display_path(app.target_vocal_path)}")
    app.cleanup_gpu()
    return app.target_vocal_path