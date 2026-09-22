import os
from pathlib import Path

import customtkinter as ctk
from dotenv import load_dotenv


def configure_ctk_theme():
	project_dir = Path(__file__).resolve().parent.parent
	themes_dir = project_dir / 'themes'
	env_path = project_dir / '.env'
	env_loaded = load_dotenv(env_path)
	mode = str(os.getenv('CTk_mode') or 'system').strip().lower()
	theme = str(os.getenv('CTk_theme') or 'default.json').strip()
	theme_path = themes_dir / theme
	default_theme_path = themes_dir / 'default.json'

	if env_loaded and theme_path.is_file():
		print(f"[Theme] Success: loaded {theme} from {env_path}", flush=True)
	else:
		print(f"[Theme] Not successful: loading default.json (env loaded: {env_loaded}, theme found: {theme_path.is_file()})", flush=True)
		theme_path = default_theme_path

	ctk.set_appearance_mode(mode if mode in {'system', 'light', 'dark'} else 'system')
	ctk.set_default_color_theme(str(theme_path))

"""
from utils.ctk_theme import configure_ctk_theme

configure_ctk_theme()
"""
