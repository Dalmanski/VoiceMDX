# VoiceMDX — VS Code Setup

## WIP! Released since Sep 11, 2026

## 1. Folder structure

```text
VoiceMDX/
├── app.py
├── ffmpeg.exe
├── themes/
│   └── default.json
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

Download and install Python 3.12:

https://www.python.org/downloads/windows/

During installation, enable:

```text
Add Python to PATH
```

Then open a new VS Code terminal:

```powershell
python --version
```

If needed:

```text
Ctrl+Shift+P → Python: Select Interpreter → Python 3.12
```

Install the Python dependencies:

```powershell
py -m pip install -r requirements.txt
```

Install PyTorch using the CUDA build recommended by your Seed-VC setup.

---

# 3. Download the required files

## FFmpeg

Download a Windows build:

https://ffmpeg.org/download.html

Extract it and copy:

```text
bin/ffmpeg.exe
```

beside `app.py`:

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

Repository:

https://github.com/Plachtaa/seed-vc

Clone it:

```powershell
git clone https://github.com/Plachtaa/seed-vc.git
```

Check it:

```powershell
py .\seed-vc\inference.py --help
```

---

## UVR models

Create:

```text
UVR_MODELS/
```

### Instrument model

Download `UVR-MDX-NET-Inst_HQ_4.onnx`:

https://huggingface.co/Politrees/UVR_resources/blob/main/models/MDXNet/UVR-MDX-NET-Inst_HQ_4.onnx

Save as:

```text
UVR_MODELS/UVR-MDX-NET-Inst_HQ_4.onnx
```

### Vocal model

Download `UVR-MDX-NET-Voc_FT.onnx`:

https://huggingface.co/Politrees/UVR_resources/blob/main/models/MDXNet/UVR-MDX-NET-Voc_FT.onnx

Save as:

```text
UVR_MODELS/UVR-MDX-NET-Voc_FT.onnx
```

---

# 4. Run

```powershell
py app.py
```

---

# 5. Phone Microphone with WO Mic (Optional)

WO Mic lets you use your phone as a microphone on your PC.

Download:

https://wolicheng.com/womic/download.html

## Setup

### 1. Install WO Mic

Install WO Mic on your phone and install the **WO Mic Client and Driver** on Windows.

### 2. Connect to the same Wi-Fi

Connect your phone and PC to the same Wi-Fi network.

### 3. Start the phone microphone

Open WO Mic on your phone, select **Wi-Fi** as the transport, and start the server.

### 4. Connect on Windows

Open **WO Mic Client**:

```text
Connection → Connect... → Transport: Wi-Fi
```

Enter the IP address shown on your phone and connect.

### 5. Select the microphone

In Windows:

```text
Settings → System → Sound → Input
```

Select:

```text
WO Mic Device
```

The setup is:

```text
Phone Mic
   ↓
WO Mic
   ↓
Wi-Fi
   ↓
WO Mic Client
   ↓
WO Mic Device
```

---

# 6. Quick checks

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
themes/default.json
```

## Troubleshooting

**FFmpeg missing:** Put `ffmpeg.exe` beside `app.py`.

**Seed-VC missing:** Make sure `seed-vc/inference.py` exists.

**UVR model missing:** Make sure both `.onnx` files are inside `UVR_MODELS`.

**CUDA = False:** Install a CUDA-enabled PyTorch build and make sure your NVIDIA driver is installed.

**WO Mic not connecting:** Make sure both devices use the same Wi-Fi and that the phone's IP address is entered correctly in WO Mic Client.

**WO Mic Device missing:** Reinstall the WO Mic Windows client and driver, then check Windows sound input devices.
