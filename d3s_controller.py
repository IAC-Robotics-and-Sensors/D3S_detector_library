"""
D3S Detector Controller
-----------------------
Thread-safe controller for the Kromek D3S detector over serial.

Features:
 - Connect / Disconnect lifecycle (matches Hamamatsu controller API)
 - Thread-safe background acquisition with daemon thread
 - Efficient bulk serial reads (single read call per frame)
 - CPS estimation via collections.deque (O(1) trim)
 - Neutron count and temperature telemetry
 - Serial number retrieval
 - Reset & flush logic
 - Fixed-duration (timed) acquisition
 - Periodic logging with buffered file I/O and live delta_t
"""

from __future__ import annotations

import io
import os
import struct
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import serial
from serial.tools import list_ports

# Frame sizes (bytes)
_SPECTRUM_FRAME_SIZE = 8205
_STATUS_FRAME_SIZE = 17
_SERIAL_FRAME_SIZE = 111

# Commands
_CMD_SPECTRUM = b"\x07\x00\x00\x07\xc1\x00\x00"
_CMD_STATUS   = b"\x07\x00\x00\x07\xc5\x00\x00"
_CMD_SERIAL   = b"\x07\x00\x00\x07\xc7\x00\x00"

# Struct formats (pre-compiled for speed)
_FMT_SPECTRUM = struct.Struct("<HB" + "BB" + "L" + "H" + "H" * 4096 + "H")
_FMT_STATUS   = struct.Struct("<HB" + "BBBB" + "b" + "BBBB" + "b" + "BB" + "H")
_FMT_SERIAL   = struct.Struct("<HB" + "BB" + "HH" + "c" * 50 + "c" * 50 + "H")

GE_TABLE_SIZE = 4096


class D3SDetector:
    """Low-level serial interface to the D3S detector."""

    def __init__(self, port: Optional[str] = None):
        self.port = port or self._find_port()
        self.ser: Optional[serial.Serial] = None
        self._open_port()

    # ── port discovery ───────────────────────────────────────────────────────

    @staticmethod
    def _find_port(timeout: float = 30.0) -> str:
        """Scan for a ttyACM device with an optional timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for p in list_ports.comports():
                if os.path.basename(p.device).startswith("ttyACM"):
                    print(f"D3S detected on {p.device}")
                    return p.device
            print("Waiting for D3S device...")
            time.sleep(1)
        raise IOError("D3S device not found within timeout")

    def _open_port(self):
        self.ser = serial.Serial(
            port=self.port,
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1,
        )
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    @property
    def is_open(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def close(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass

    def reconnect(self):
        self.close()
        self.port = self._find_port()
        self._open_port()

    # ── data reads ───────────────────────────────────────────────────────────

    def read_spectrum(self) -> Tuple[np.ndarray, int]:
        """Request and read one spectrum frame.

        Returns (4096-channel spectrum, neutron_count).
        Uses a single bulk read for efficiency.
        """
        try:
            self.ser.reset_input_buffer()
            self.ser.write(_CMD_SPECTRUM)

            # Bulk read — much faster than byte-by-byte
            data = self.ser.read(_SPECTRUM_FRAME_SIZE)
            if len(data) < _SPECTRUM_FRAME_SIZE:
                raise IOError(
                    f"Incomplete spectrum frame ({len(data)}/{_SPECTRUM_FRAME_SIZE})"
                )

            dataf = _FMT_SPECTRUM.unpack(data)
            spectrum = np.array(dataf[6:-1], dtype=np.uint32)
            neutron_count = dataf[5]
            return spectrum, neutron_count
        except Exception as exc:
            print(f"Serial read error: {exc} — reconnecting...")
            self.reconnect()
            return np.zeros(GE_TABLE_SIZE, dtype=np.uint32), 0

    def read_status(self) -> Tuple[float, int]:
        """Read temperature and raw status from the D3S.

        Returns (temperature_celsius, raw_scan_time).
        """
        try:
            self.ser.reset_input_buffer()
            self.ser.write(_CMD_STATUS)
            data = self.ser.read(_STATUS_FRAME_SIZE)
            if len(data) < _STATUS_FRAME_SIZE:
                raise IOError("Incomplete status frame")
            fields = _FMT_STATUS.unpack(data)
            temperature = float(fields[6])
            scan_time = fields[4]
            return temperature, scan_time
        except Exception as exc:
            print(f"Status read error: {exc}")
            return float("nan"), 0

    def read_serial_number(self) -> str:
        """Read the detector serial number string."""
        try:
            self.ser.reset_input_buffer()
            self.ser.write(_CMD_SERIAL)
            data = self.ser.read(_SERIAL_FRAME_SIZE)
            if len(data) < _SERIAL_FRAME_SIZE:
                raise IOError("Incomplete serial frame")
            fields = _FMT_SERIAL.unpack(data)
            serial_bytes = b"".join(fields[56:65])
            return serial_bytes.decode("utf-8", errors="replace").strip("\x00")
        except Exception as exc:
            print(f"Serial-number read error: {exc}")
            return "unknown"


class D3SController:
    """High-level threaded controller for the Kromek D3S detector.

    Mirrors the HamamatsuController API so both GUIs share the same
    connect / start / stop / disconnect lifecycle.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.detector: Optional[D3SDetector] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()

        # Spectrum & timing
        self.spectrum = np.zeros(GE_TABLE_SIZE, dtype=np.uint32)
        self.elapsed_time = 0.0
        self._start_time: Optional[float] = None

        # CPS estimation — deque is O(1) for left-trim vs list comprehension
        self.cps = 0.0
        self._cps_window = 3.0  # seconds
        self._history: deque = deque()

        # Telemetry
        self.temperature = float("nan")
        self.neutron_count = 0
        self.serial_number = ""

        # Logging
        self._log_thread: Optional[threading.Thread] = None
        self._log_stop_event = threading.Event()
        self._log_filename: Optional[str] = None
        self._log_interval: Optional[float] = None
        self._log_total_time: Optional[float] = None
        self.last_delta_t: Optional[float] = None

    # --------------------------------------------------------
    # CONNECTION (matches Hamamatsu API)
    # --------------------------------------------------------

    def connect(self) -> bool:
        """Find and open the D3S serial device."""
        try:
            self.detector = D3SDetector()
            self.serial_number = self.detector.read_serial_number()
            if self.verbose:
                print(f"D3S connected (S/N: {self.serial_number})")
            return True
        except Exception as e:
            print(f"Error connecting to D3S: {e}")
            self.detector = None
            return False

    def disconnect(self):
        """Stop acquisition and close the serial port."""
        if self._running:
            self.stop()
        if self.detector:
            self.detector.close()
            self.detector = None
        if self.verbose:
            print("D3S disconnected.")

    @property
    def is_connected(self) -> bool:
        return self.detector is not None and self.detector.is_open

    # --------------------------------------------------------
    # START / STOP
    # --------------------------------------------------------

    def start(self):
        """Start background acquisition."""
        if self._running:
            return
        if not self.is_connected:
            if not self.connect():
                return
        if self.verbose:
            print("Starting D3SController...")
        self._stop_event.clear()
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print("Acquisition started.")

    def stop(self):
        """Stop acquisition and logging safely."""
        if not self._running:
            return
        if self.verbose:
            print("Stopping acquisition...")
        self._stop_event.set()
        self.stop_periodic_logging()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._running = False
        if self.verbose:
            print("Acquisition stopped cleanly.")

    # --------------------------------------------------------
    # ACQUISITION LOOP
    # --------------------------------------------------------

    def _acquisition_loop(self):
        while not self._stop_event.is_set():
            try:
                spectrum, neutrons = self.detector.read_spectrum()
                now = time.time()
                with self._lock:
                    self.spectrum += spectrum
                    self.neutron_count += neutrons
                    self.elapsed_time = now - self._start_time

                    # CPS via deque — O(1) left-trim
                    total_counts = int(self.spectrum.sum())
                    self._history.append((now, total_counts))
                    cutoff = now - self._cps_window
                    while self._history and self._history[0][0] < cutoff:
                        self._history.popleft()

                    if len(self._history) > 1:
                        dt = self._history[-1][0] - self._history[0][0]
                        if dt > 0:
                            dc = self._history[-1][1] - self._history[0][1]
                            self.cps = dc / dt
            except Exception as e:
                print(f"Acquisition error: {e}")
                break
            time.sleep(0.02)  # ~50 Hz polling (was 0.05)
        if self.verbose:
            print("Acquisition loop exited cleanly.")

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    def read_temperature(self) -> float:
        """Read the D3S internal temperature (blocking)."""
        if not self.is_connected:
            return float("nan")
        temp, _ = self.detector.read_status()
        self.temperature = temp
        return temp

    # --------------------------------------------------------
    # DATA CONTROL
    # --------------------------------------------------------

    def reset(self):
        """Reset cumulative spectrum, timer, CPS, and history."""
        with self._lock:
            self.spectrum[:] = 0
            self._start_time = time.time()
            self.elapsed_time = 0.0
            self.cps = 0.0
            self.neutron_count = 0
            self._history.clear()
        if self.verbose:
            print("Spectrum reset.")

    def get_spectrum(self) -> Tuple[np.ndarray, float, float]:
        """Return (spectrum_copy, elapsed_time, cps).

        Automatically starts acquisition if not already running.
        """
        if not self._running:
            if self.verbose:
                print("Acquisition not running — starting automatically.")
            self.start()
            time.sleep(0.5)
        with self._lock:
            return np.copy(self.spectrum), self.elapsed_time, self.cps

    # --------------------------------------------------------
    # TIMED ACQUISITION
    # --------------------------------------------------------

    def acquire_spectrum_for_duration(
        self, duration: float, filename: Optional[str] = None
    ) -> Tuple[np.ndarray, float]:
        """Acquire and return a spectrum for a fixed duration."""
        if not self._running:
            self.start()
        self.reset()
        start = time.time()
        while time.time() - start < duration and self._running:
            time.sleep(0.1)
        spec, elapsed, _ = self.get_spectrum()
        if filename:
            np.savetxt(filename, spec)
            if self.verbose:
                print(f"Spectrum ({elapsed:.1f}s) saved to {filename}")
        return spec, elapsed

    # --------------------------------------------------------
    # PERIODIC LOGGING (buffered I/O)
    # --------------------------------------------------------

    def start_periodic_logging(
        self, base_filename: str, interval: float = 10.0, total_time: float = 0.0
    ):
        """Start logging cumulative spectrum every *interval* seconds."""
        if self._log_thread and self._log_thread.is_alive():
            if self.verbose:
                print("Logging already active.")
            return

        self.reset()
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(base_filename)
        if ext == "":
            ext = ".csv"
        filename = f"{root}_{dt_str}{ext}"

        self._log_filename = filename
        self._log_interval = interval
        self._log_total_time = total_time
        self._log_stop_event.clear()
        self.last_delta_t = 0.0

        # Write header
        with open(filename, "w") as f:
            f.write("delta_t," + ",".join(f"ch{i}" for i in range(GE_TABLE_SIZE)) + "\n")

        self._log_thread = threading.Thread(target=self._logging_loop, daemon=True)
        self._log_thread.start()
        if self.verbose:
            dur = "indefinitely" if total_time == 0 else f"for {total_time}s"
            print(f"Periodic logging started ({interval}s interval, {dur}) -> {filename}")

    def _logging_loop(self):
        prev_time = time.time()
        start_time = prev_time

        while not self._log_stop_event.is_set():
            now = time.time()
            self.last_delta_t = now - prev_time
            prev_time = now

            spectrum, _, _ = self.get_spectrum()

            # Buffered write — build the full line in memory, then flush once
            buf = io.StringIO()
            buf.write(f"{self.last_delta_t:.3f},")
            buf.write(",".join(map(str, spectrum)))
            buf.write("\n")
            try:
                with open(self._log_filename, "a") as f:
                    f.write(buf.getvalue())
            except Exception as e:
                print(f"Error writing log file: {e}")

            if self._log_total_time > 0 and (now - start_time) >= self._log_total_time:
                break
            self._log_stop_event.wait(timeout=self._log_interval)

        if self.verbose:
            print("Logging loop ended.")

    def stop_periodic_logging(self):
        """Stop periodic logging if running."""
        if self._log_thread and self._log_thread.is_alive():
            self._log_stop_event.set()
            self._log_thread.join(timeout=2.0)
            if self.verbose:
                print("Periodic logging stopped.")
