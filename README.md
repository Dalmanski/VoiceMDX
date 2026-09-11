# VoiceMDX — VS Code Setup

## WIP! Released since Sep 11,2026 

## 1. Folder structure

```text
VoiceMDX/
├── app.py
├── ffmpeg.exe
├── themes/
│   └── custom.json
├── UVR_MODELS/
│   ├── UVR-MDX-NET-Inst_HQ_4.onnx
│   └── UVR-MDX-NET-Voc_FT.onnx
└── seed-vc/
    ├── inference.py
    └── modules/
        └── bigvgan/
            └── bigvgan.py
```

## 2. Install Python 3.12

Download and install Python 3.12 from the official Python website:

https://www.python.org/downloads/windows/

During installation, make sure to enable:

```text
Add Python to PATH
```

Then open a new VS Code terminal and verify it is using Python 3.12:

```powershell
python --version
```

If needed, in VS Code choose:

```text
Ctrl+Shift+P → Python: Select Interpreter → Python 3.12
```

Install the basic packages:

```powershell
python -m pip install --upgrade pip
pip install customtkinter audio-separator
```

Install PyTorch using the CUDA build recommended by your Seed-VC setup.

---

# 3. Download the required files

## FFmpeg

Download a Windows build from the official FFmpeg download page:

https://ffmpeg.org/download.html

Under **Windows EXE Files**, use a Windows build such as Gyan.dev or BtbN. citeturn585035search0

After extracting the ZIP:

```text
bin/ffmpeg.exe
```

Copy `ffmpeg.exe` beside `app.py`:

```text
VoiceMDX/
├── app.py
└── ffmpeg.exe
```

Test it:

```powershell
.\ffmpeg.exe -version
```

---

## Seed-VC

Use the Seed-VC GitHub repository:

https://github.com/Plachtaa/seed-vc citeturn585035search5

Easiest method:

```powershell
git clone https://github.com/Plachtaa/seed-vc.git
```

This creates:

```text
VoiceMDX/
└── seed-vc/
```

Install its dependencies:

```powershell
pip install -r .\seed-vc\requirements.txt
```

The repository contains `inference.py`, which your app runs. citeturn585035search1turn585035search5

Check:

```powershell
python .\seed-vc\inference.py --help
```

---

## UVR models

Create:

```text
UVR_MODELS/
```

### Instrument model

Download:

`UVR-MDX-NET-Inst_HQ_4.onnx`

Source:

https://huggingface.co/Politrees/UVR_resources/blob/main/models/MDXNet/UVR-MDX-NET-Inst_HQ_4.onnx citeturn296991search2

Click **Download** and save it as:

```text
UVR_MODELS/UVR-MDX-NET-Inst_HQ_4.onnx
```

### Vocal model

Download:

`UVR-MDX-NET-Voc_FT.onnx`

Source:

https://huggingface.co/Politrees/UVR_resources/blob/main/models/MDXNet/UVR-MDX-NET-Voc_FT.onnx citeturn585035search7

Click **Download** and save it as:

```text
UVR_MODELS/UVR-MDX-NET-Voc_FT.onnx
```

---

# 4. Run

Make sure your terminal shows:

```text
(.venv)
```

Then:

```powershell
python app.py
```

---

# 5. Quick checks

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from audio_separator.separator import Separator; print('audio-separator OK')"
.\ffmpeg.exe -version
python .\seed-vc\inference.py --help
```

Make sure these exist:

```text
ffmpeg.exe
seed-vc/inference.py
UVR_MODELS/UVR-MDX-NET-Inst_HQ_4.onnx
UVR_MODELS/UVR-MDX-NET-Voc_FT.onnx
themes/custom.json
```

## Troubleshooting

**FFmpeg missing:** put `ffmpeg.exe` beside `app.py`.

**Seed-VC missing:** make sure `seed-vc/inference.py` exists.

**UVR model missing:** make sure both `.onnx` files are inside `UVR_MODELS`.

**CUDA = False:** install a CUDA-enabled PyTorch build and make sure your NVIDIA driver is installed.
