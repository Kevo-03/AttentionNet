# AttentionNet Demo Application

A comprehensive demonstration system for the AttentionNet network traffic classifier.

## Four Demo Options

### Option 1: Streamlit Web App (`demo_streamlit.py`)
Beautiful web interface that runs in your browser. No desktop GUI dependencies.

### Option 2: PyQt6 Desktop App (`demo_pyqt.py`) ⭐ NEW
Professional desktop GUI with native look. Works with pyenv (no Tkinter issues)!

### Option 3: CLI Application (`demo_cli.py`)
Command-line interface - works everywhere, no GUI dependencies.

### Option 4: Tkinter GUI (`demo_app.py`)
Desktop GUI - requires Tkinter (may need extra setup on macOS with pyenv).

## Features

1. **🔴 Live Capture** - Capture packets from a network interface and classify in real-time
2. **📁 Load PCAP** - Process and classify traffic from PCAP/PCAPNG files
3. **📊 Test Dataset** - Evaluate the model on pre-processed test datasets
4. **ℹ️ About** - Model architecture and usage information

## Requirements

```bash
pip install torch numpy scikit-learn matplotlib scapy streamlit pandas
```

**Note:** Scapy is optional but required for PCAP processing.

## Usage

### Option 1: Streamlit Web App

```bash
cd /Users/kivanc/Desktop/AttentionNet
source venv/bin/activate
streamlit run demo/demo_streamlit.py
```

Opens a browser window with the demo.

### Option 2: PyQt6 Desktop App ⭐ RECOMMENDED

```bash
cd /Users/kivanc/Desktop/AttentionNet
source venv/bin/activate
python demo/demo_pyqt.py
```

Opens a native desktop window. For live capture:
```bash
sudo python demo/demo_pyqt.py
```

### Option 3: CLI Application

```bash
cd /Users/kivanc/Desktop/AttentionNet
source venv/bin/activate
python demo/demo_cli.py
```

### Option 4: Tkinter GUI (if Tkinter is available)

```bash
python demo/demo_app.py
```

### Fixing Tkinter on macOS with pyenv

If you get `ModuleNotFoundError: No module named '_tkinter'`, you need to reinstall Python with Tkinter support:

```bash
# Install tcl-tk first
brew install tcl-tk

# Set environment variables for pyenv to find tcl-tk
export LDFLAGS="-L$(brew --prefix tcl-tk)/lib"
export CPPFLAGS="-I$(brew --prefix tcl-tk)/include"
export PKG_CONFIG_PATH="$(brew --prefix tcl-tk)/lib/pkgconfig"
export PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I$(brew --prefix tcl-tk)/include' --with-tcltk-libs='-L$(brew --prefix tcl-tk)/lib -ltcl8.6 -ltk8.6'"

# Reinstall Python
pyenv install 3.11.5
```

Alternatively, use the system Python which usually has Tkinter:
```bash
/usr/bin/python3 demo/demo_app.py
```

### Live Capture (requires root/admin)

For live network capture, you may need elevated privileges:

**macOS/Linux:**
```bash
sudo python demo_app.py
```

**Windows:**
Run Command Prompt as Administrator, then run the script.

### Common Network Interfaces

- **macOS:** `en0` (WiFi), `en1` (Ethernet)
- **Linux:** `eth0`, `wlan0`, `enp3s0`
- **Windows:** Use interface name from `ipconfig`

## Tabs Overview

### 🔴 Live Capture Tab
- Enter your network interface name
- Set capture duration (seconds)
- Click "Start Capture"
- View classified flows with visualizations

### 📁 Load PCAP Tab
- Browse for a .pcap or .pcapng file
- Set maximum flows to process
- Click "Process & Classify"
- View results and probability distributions

### 📊 Test Dataset Tab
- Load pre-processed test/train .npy files
- Run evaluation to see accuracy metrics
- View confusion matrix
- Export classification report

### ℹ️ About Tab
- Model architecture details
- Classification classes
- Processing pipeline explanation

## Screenshots

The application provides:
- 28×28 flow image visualizations
- Per-flow classification results
- Confidence scores and probability distributions
- Confusion matrices for dataset evaluation
- Exportable classification reports

## Troubleshooting

### "Scapy not available"
Install scapy: `pip install scapy`

### "Permission denied" during live capture
Run with sudo/admin privileges

### "Model not found"
Make sure the model file exists at:
`model_output/memory_safe/hocaya_gosterilcek/p2p_change/2layer_cnn_hybrid_3fc/best_model.pth`

### "No valid flows found in PCAP"
The PCAP file must contain IP packets with TCP or UDP transport layer.

