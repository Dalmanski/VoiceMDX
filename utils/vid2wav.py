import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FFMPEG = BASE_DIR / "ffmpeg.exe"
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".caf"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".ts", ".mts", ".m2ts"]
ALL_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS


def find_ffmpeg(ffmpeg_path=None):
    if ffmpeg_path:
        candidate = Path(ffmpeg_path)
        if candidate.exists():
            return str(candidate)
    if DEFAULT_FFMPEG.exists():
        return str(DEFAULT_FFMPEG)
    return shutil.which("ffmpeg")


def is_video_file(path):
    suffix = Path(path).suffix.lower()
    return suffix in VIDEO_EXTENSIONS


def convert_media_to_wav(input_path, output_path, ffmpeg_path=None, channels=2, sample_rate=44100, label="input"):
    ffmpeg = find_ffmpeg(ffmpeg_path)
    if not ffmpeg:
        raise RuntimeError(f"FFmpeg was not found: {DEFAULT_FFMPEG}")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [ffmpeg, "-y", "-i", str(input_path), "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(output_path)]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

    if completed.returncode != 0 or not output_path.exists():
        if completed.stdout:
            for line in completed.stdout.splitlines():
                if line.strip():
                    print(line)
        raise RuntimeError(f"FFmpeg failed to convert {label} to WAV.")

    return output_path


def extract_audio(input_path, output_path, ffmpeg_path=None, channels=1, sample_rate=44100, label="input"):
    return convert_media_to_wav(input_path, output_path, ffmpeg_path=ffmpeg_path, channels=channels, sample_rate=sample_rate, label=label)


def convert_video_to_wav(input_path, output_path, ffmpeg_path=None, channels=2, sample_rate=44100):
    return convert_media_to_wav(input_path, output_path, ffmpeg_path=ffmpeg_path, channels=channels, sample_rate=sample_rate, label="video")

