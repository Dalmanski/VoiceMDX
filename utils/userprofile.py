import os
import re
import sys
import tempfile
from pathlib import Path


def display_user_path(path, temp_root=None):
	try:
		raw = str(path)
		if not raw:
			return raw
		candidate = os.path.normpath(raw)
		if sys.platform.startswith("win"):
			try:
				import ctypes
				long_path = ctypes.create_unicode_buffer(32768)
				if ctypes.windll.kernel32.GetLongPathNameW(candidate, long_path, len(long_path)):
					candidate = long_path.value
			except Exception:
				pass
		candidate = os.path.abspath(candidate)
		temp_root = os.path.normpath(str(temp_root or (Path(tempfile.gettempdir()) / "voicemdx_temp")))
		user_profile = os.path.normpath(os.environ.get("USERPROFILE", str(Path.home())))
		try:
			rel = os.path.relpath(candidate, temp_root)
			if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
				return os.path.join("%USERPROFILE%", "AppData", "Local", "Temp", "voicemdx_temp", rel)
		except (ValueError, OSError):
			pass
		try:
			rel = os.path.relpath(candidate, user_profile)
			if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
				return os.path.join("%USERPROFILE%", rel)
		except (ValueError, OSError):
			pass
		return raw
	except Exception:
		return str(path)


def display_user_paths(text):
	if not isinstance(text, str):
		return text

	def replace_path(match):
		raw = match.group(0)
		suffix = ""
		while raw and raw[-1] in ".,;:)]}\"'":
			suffix = raw[-1] + suffix
			raw = raw[:-1]
		return display_user_path(raw) + suffix

	return re.sub(r"(?i)[A-Z]:\\[^\r\n]+", replace_path, text)
