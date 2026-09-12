import os

OFFLINE_MODE = os.getenv("OFFLINE_MODE", "1") == "1"

def enable_offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

def disable_offline_mode():
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("DIFFUSERS_OFFLINE", None)

def configure():
    if OFFLINE_MODE:
        enable_offline_mode()
    else:
        disable_offline_mode()

configure()