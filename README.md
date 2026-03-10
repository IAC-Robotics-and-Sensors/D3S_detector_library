# D3S Detector Control & GUI

A Python interface for collecting gamma-ray spectra from **Kromek D3S** radiation detectors via USB serial connection. This project is designed to be used on Linux systems and provides both a programmatic API and an interactive GUI for live monitoring and data acquisition.

This project provides:
- A **thread-safe controller class (`D3SController`)** for data acquisition.
- A **PyQt6 + Matplotlib GUI (`d3s_gui.py`)** for live monitoring, timed acquisition, periodic logging, and energy calibration.
- Example scripts for programmatic use.

---

## Features

- **Automatic serial port detection** (`/dev/ttyACM*` on Linux)
- **Connect / Disconnect lifecycle** with serial number display
- **Continuous background acquisition** with daemon threads
- **Efficient bulk serial reads** (single read per frame, pre-compiled struct formats)
- **Real-time counts-per-second (CPS)** via sliding-window deque
- **Neutron count tracking**
- **Temperature reading** from the D3S status packet
- **Timed (fixed-duration) acquisitions**
- **Periodic logging** to timestamped CSVs with per-spectrum `Δt` and buffered I/O
- **Log scale toggle** for the spectrum plot
- **Energy calibration** — interactive marker placement, polynomial fit, save/load calibration as JSON
- **Thread-safe and clean shutdown**
- **Tabbed PyQt6 GUI** (Spectrum, Acquisition, Calibration, Device Info)

---

## Project Structure

```
.
├── d3s_controller.py      # Main threaded detector controller
├── d3s_gui.py             # PyQt6 GUI with live spectrum & calibration
├── d3sLibrary.py          # Legacy multiprocessing-based D3S library
├── example_acquisition.py # Example script for timed/periodic runs
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Installation

It is recommended to use a virtual environment with Python 3.10+.

```bash
pip install -r requirements.txt
```

Dependencies: `pyserial`, `numpy`, `matplotlib`, `PyQt6`

---

## Usage

### GUI

Run the live graphical interface:

```bash
python d3s_gui.py
```

**Tabs and features:**

| Tab | Features |
|-----|----------|
| **Spectrum** | Live plot, Start/Stop/Clear/Save, log scale toggle, CPS, neutron count, elapsed time |
| **Acquisition** | Timed acquisition (fixed duration), periodic logging (interval + total time) |
| **Calibration** | Interactive marker placement on spectrum, polynomial fit (order 1–5), apply to x-axis, save/load calibration JSON |
| **Device Info** | Serial number display, temperature reading |

**Workflow:**
1. Click **Connect** to find and open the D3S device.
2. Click **Start Acquisition** to begin collecting spectra.
3. Use the **Calibration** tab to map channels to energy (keV).
4. Use **Save Spectrum** or **Periodic Logging** to export data.
5. Click **Stop**, then **Disconnect** when finished.

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

### Calibration
`d3s_calibration_YYYYMMDD_HHMMSS.json`
→ JSON file containing marker points, polynomial order, and fitted coefficients.

---

## API Quick Reference

### Class: `D3SController`

| Method / Property | Description |
|-------------------|-------------|
| `connect()` | Find and open the D3S serial device; returns `True` on success |
| `disconnect()` | Stop acquisition and close serial port |
| `is_connected` | Property: whether the device is open |
| `start()` | Begin background acquisition thread |
| `stop()` | Stop acquisition and logging |
| `reset()` | Reset cumulative spectrum, timer, CPS, and neutron count |
| `get_spectrum()` | Return `(spectrum, elapsed_time, cps)` |
| `acquire_spectrum_for_duration(duration, filename=None)` | Timed acquisition |
| `start_periodic_logging(base_filename, interval, total_time=0)` | Periodic CSV logging |
| `stop_periodic_logging()` | Stop ongoing logging |
| `read_temperature()` | Read and return the D3S internal temperature (°C) |
| `serial_number` | Device serial number string (populated on connect) |
| `neutron_count` | Cumulative neutron count since last reset |
| `cps` | Current counts per second (sliding window) |
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