from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.vid2wav import convert_media_to_wav

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


@dataclass(frozen=True)
class SourceStems:
    vocal_path: Path
    instrumental_path: Path | None
    has_background: bool

def validate():
    if Separator is None:
        raise RuntimeError(f"audio-separator import failed: {type(AUDIO_SEPARATOR_IMPORT_ERROR).__name__}: {AUDIO_SEPARATOR_IMPORT_ERROR}")
    if not UVR_VOCAL_MODEL.exists():
        raise RuntimeError(f"Missing vocal model: {UVR_VOCAL_MODEL}")

def create_separator(output_dir, stem_name, batch_size=UVR_BATCH_SIZE, log=print):
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
    log(f"UVR CUDA: CUDAExecutionProvider | batch={batch_size}")
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

def match_source_volume(source_path, vocal_path):
    source, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    vocal, vocal_rate = sf.read(vocal_path, dtype="float32", always_2d=True)
    if sample_rate != vocal_rate or source.size == 0 or vocal.size == 0:
        return vocal_path
    length = min(len(source), len(vocal))
    source_mono = source[:length].mean(axis=1)
    vocal_mono = vocal[:length].mean(axis=1)
    projection = np.dot(source_mono, vocal_mono) / max(np.dot(vocal_mono, vocal_mono), 1e-12)
    reference = vocal_mono * projection
    source_rms = float(np.sqrt(np.mean(reference ** 2)))
    vocal_rms = float(np.sqrt(np.mean(vocal_mono ** 2)))
    if source_rms <= 0 or vocal_rms <= 0:
        return vocal_path
    source_db = 20 * np.log10(source_rms)
    separation_db = 20 * np.log10(vocal_rms)
    gain_db = source_db - separation_db
    print(f"Source vocal reference volume: {source_db:+.2f} dBFS")
    print(f"Separated vocal volume: {separation_db:+.2f} dBFS")
    print(f"Volume adjustment: {separation_db:+.2f} + ({source_db:+.2f} - {separation_db:+.2f}) = {source_db:+.2f} dBFS | gain={gain_db:+.2f} dB")
    matched = 0.98 * np.tanh(vocal * (10 ** (gain_db / 20)) / 0.98)
    sf.write(vocal_path, matched, vocal_rate, subtype="PCM_16")
    print(f"Separated vocal volume after match: {20 * np.log10(max(np.sqrt(np.mean(matched ** 2)), 1e-12)):+.2f} dBFS")
    return vocal_path

def match_instrumental_volume(source_path, vocal_path, instrumental_path):
    source, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    vocal, vocal_rate = sf.read(vocal_path, dtype="float32", always_2d=True)
    instrumental, instrumental_rate = sf.read(instrumental_path, dtype="float32", always_2d=True)
    if sample_rate != vocal_rate or sample_rate != instrumental_rate:
        return instrumental_path
    length = min(len(source), len(vocal))
    source_mono = source[:length].mean(axis=1)
    vocal_mono = vocal[:length].mean(axis=1)
    projection = np.dot(source_mono, vocal_mono) / max(np.dot(vocal_mono, vocal_mono), 1e-12)
    reference = source_mono - vocal_mono * projection
    reference_rms = float(np.sqrt(np.mean(reference ** 2)))
    instrumental_rms = float(np.sqrt(np.mean(instrumental ** 2)))
    if reference_rms <= 0 or instrumental_rms <= 0:
        return instrumental_path
    reference_db = 20 * np.log10(reference_rms)
    instrumental_db = 20 * np.log10(instrumental_rms)
    gain_db = reference_db - instrumental_db
    print(f"Source instrumental reference volume: {reference_db:+.2f} dBFS")
    print(f"Separated instrumental volume: {instrumental_db:+.2f} dBFS")
    print(f"Instrumental adjustment: {instrumental_db:+.2f} + ({reference_db:+.2f} - {instrumental_db:+.2f}) = {reference_db:+.2f} dBFS | gain={gain_db:+.2f} dB")
    matched = 0.98 * np.tanh(instrumental * (10 ** (gain_db / 20)) / 0.98)
    sf.write(instrumental_path, matched, instrumental_rate, subtype="PCM_16")
    print(f"Separated instrumental volume after match: {20 * np.log10(max(np.sqrt(np.mean(matched ** 2)), 1e-12)):+.2f} dBFS")
    return instrumental_path

def separate(input_path, model_path, output_dir, stem_name, retry_message, missing_message, log=print, cleanup_gpu=lambda: None):
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.glob("*.wav"):
        try:
            item.unlink()
        except Exception:
            pass
    log(f"Loading {model_path.name}")
    separator = create_separator(output_dir, stem_name, log=log)
    try:
        result = separator.separate(str(input_path))
    except Exception as exc:
        if UVR_BATCH_SIZE <= 1 or "memory" not in str(exc).lower():
            raise
        log(retry_message)
        cleanup_gpu()
        result = create_separator(output_dir, stem_name, 1, log=log).separate(str(input_path))
    paths = flatten_paths(result)
    discovered = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    paths += [path for path in discovered if path not in paths]
    selected = pick_stem_path(paths, stem_name) or (paths[0] if paths else None)
    if not selected or not selected.exists():
        raise RuntimeError(missing_message)
    return selected

def ensure_source_stems(source_path, ffmpeg_path, inputs_dir, vocal_dir, inst_dir, log=print, cleanup_gpu=lambda: None):
    if not source_path or not Path(source_path).exists():
        raise RuntimeError("Source audio or video was not found.")
    if not ffmpeg_path:
        raise RuntimeError(f"FFmpeg was not found: {ffmpeg_path}")
    validate()
    source_wav = Path(inputs_dir) / "source.wav"
    convert_media_to_wav(source_path, source_wav, ffmpeg_path=ffmpeg_path, channels=2, sample_rate=44100, label="source")
    vocal_path = separate(source_wav, UVR_VOCAL_MODEL, vocal_dir, "Vocals", "UVR batch 2 memory error; retrying with batch 1...", f"{UVR_VOCAL_MODEL.name} did not produce Vocals.wav", log, cleanup_gpu)
    cleanup_gpu()
    background = has_background(source_wav, vocal_path)
    print(f"Source background detected: {'yes' if background else 'no'}")
    vocal_path = match_source_volume(source_wav, vocal_path)
    instrumental_path = None
    if background:
        if not UVR_INSTRUMENT_MODEL.exists():
            raise RuntimeError(f"Missing instrument model: {UVR_INSTRUMENT_MODEL}")
        instrumental_path = separate(source_wav, UVR_INSTRUMENT_MODEL, inst_dir, "Instrumental", "UVR batch 2 memory error; retrying with batch 1...", f"{UVR_INSTRUMENT_MODEL.name} did not produce Instrumental.wav", log, cleanup_gpu)
        cleanup_gpu()
        instrumental_path = match_instrumental_volume(source_wav, vocal_path, instrumental_path)
    if instrumental_path:
        log(f"Instrumental: {instrumental_path}")
    else:
        log("Instrumental: not available (no instrument/BG detected)")
    log(f"Source vocal: {vocal_path}")
    return SourceStems(vocal_path, instrumental_path, background)

def run_target_uvr(target_path, ffmpeg_path, inputs_dir, target_vocal_dir, log=print, cleanup_gpu=lambda: None):
    if not target_path or not Path(target_path).exists():
        raise RuntimeError("Target voice file not found.")
    if not ffmpeg_path:
        raise RuntimeError(f"FFmpeg was not found: {ffmpeg_path}")
    validate()
    target_input = Path(inputs_dir) / "target.wav"
    convert_media_to_wav(target_path, target_input, ffmpeg_path=ffmpeg_path, channels=1, sample_rate=44100, label="target")
    selected = separate(target_input, UVR_VOCAL_MODEL, target_vocal_dir, "Vocals", "UVR batch 2 memory error; retrying target with batch 1...", "Voc_FT did not produce a cleaned target vocal.", log, cleanup_gpu)
    log(f"Clean target vocal: {selected}")
    cleanup_gpu()
    return selected