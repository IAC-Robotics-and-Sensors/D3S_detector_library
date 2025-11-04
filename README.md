# D3S Detector Control & GUI

A Python interface for collecting gamma-ray spectra from **Kromek D3S** radiation detectors via USB serial connection. This project is designed to be used on Linux systems and provides both a programmatic API and an interactive GUI for live monitoring and data acquisition.

This project provides:
- A **thread-safe controller class (`D3SController`)** for data acquisition.
- A **Tkinter + Matplotlib GUI (`d3s_gui.py`)** for live monitoring, timed acquisition, and periodic logging.
- Example scripts for programmatic use.

---

## Features

- **Automatic serial port detection** (`/dev/ttyACM*` on Linux)
- **Continuous background acquisition**
- **Real-time counts-per-second (CPS)**
- **Timed (fixed-duration) acquisitions**
- **Periodic logging** to timestamped CSVs with per-spectrum `Δt`
- **Thread-safe and clean shutdown**
- **GUI with live updating spectrum**

---

## Project Structure

```
.
├── d3s_controller.py      # Main threaded detector controller
├── d3s_gui.py             # Interactive GUI with live spectrum
├── example_acquisition.py # Example script for timed/periodic runs
├── requirements.txt      # Python dependencies
├── Dockerfile             # Optional: container for deployment
└── README.md
```

---

## Installation
It is recommended to use a virtual environment with Python 3.11+.

```bash
pip install - r requirements.txt

```

To install tkinter on Debian/Ubuntu:

```bash
sudo apt-get install python3-tk
```

---

## Usage

### GUI

Run the live graphical interface:

```bash
python d3s_gui.py
```

**Features:**
- Start / Stop acquisition
- Reset detector buffer
- Save spectrum
- Timed acquisition (fixed duration)
- Periodic logging (interval, total time, Δt tracking)
- Live plot with CPS

---

### Command-Line Example

Run the provided example to see both timed and periodic acquisitions with live plots:

```bash
python example_acquisition.py
```

**This performs:**
1. A 10-second timed acquisition (saved as `example_timed_spectrum.txt`).
2. A 1-minute periodic acquisition logging every 15 seconds
   (saved as a timestamped CSV file).

---

## Output Files

### Timed Spectrum
`spectrum_YYYYMMDD_HHMMSS.txt`  
→ 4096 counts, one per channel.

### Periodic Log
`log_YYYYMMDD_HHMMSS.csv`  
→ Contains cumulative spectra over time:

```
delta_t,ch0,ch1,...,ch4095
10.002,0,1,2,1,...
9.998,5,3,2,4,...
```

---

## API Quick Reference

### Class: `D3SController`

| Method | Description |
|--------|-------------|
| `start()` | Connect and begin background acquisition |
| `stop()` | Stop acquisition and close serial port |
| `reset()` | Reset cumulative spectrum and timer |
| `get_spectrum()` | Return `(spectrum, elapsed_time, cps)` |
| `acquire_spectrum_for_duration(duration, filename=None)` | Timed acquisition |
| `start_periodic_logging(base_filename, interval, total_time=0)` | Periodic CSV logging |
| `stop_periodic_logging()` | Stop ongoing logging |
| `cps` | Current counts per second |
| `last_delta_t` | Last Δt between logged spectra |

---

## Docker Usage (Optional)

To run inside a container with serial access:

```bash
docker build -t d3s-controller .
docker run -it --device=/dev/ttyACM0 d3s-controller
```

Then inside the container:

```bash
python d3s_gui.py
```

---

## Author
Sam Fearn

Developed for radiation detection research using Kromek D3S and similar spectroscopic detectors.  