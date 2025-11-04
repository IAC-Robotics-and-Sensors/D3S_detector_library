"""
D3S Detector Controller (final)
-------------------------------
Features:
 - Thread-safe background acquisition
 - Reset & flush logic handled by GUI
 - CPS calculation from recent spectra
 - Periodic logging with live delta_t reporting
"""

import os
import time
import struct
import threading
import numpy as np
import serial
from serial.tools import list_ports
from datetime import datetime


class D3SDetector:
    """Low-level serial interface to the D3S detector."""

    def __init__(self):
        self.port = self._find_port()
        self.ser = self._open_port(self.port)

    def _find_port(self):
        while True:
            for p in list_ports.comports():
                if os.path.basename(p.device).startswith("ttyACM"):
                    print(f"D3S detected on {p.device}")
                    return p.device
            print("Waiting for D3S device...")
            time.sleep(1)

    @staticmethod
    def _open_port(port):
        ser = serial.Serial(
            port=port,
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1,
        )
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        return ser

    def reconnect(self):
        self.port = self._find_port()
        self.ser = self._open_port(self.port)

    def read_spectrum(self):
        """Request and read one spectrum frame."""
        try:
            self.ser.write(b"\x07\x00\x00\x07\xc1\x00\x00")
            data = b""
            while len(data) < 8205:
                chunk = self.ser.read(1)
                if not chunk:
                    break
                data += chunk
            if len(data) < 8205:
                raise IOError("Incomplete spectrum frame")
            dataf = struct.unpack("<HB" + "B"*2 + "L" + "H" + "H"*4096 + "H", data)
            spectrum = np.array(dataf[6:-1], dtype=np.uint32)
            return spectrum, dataf[5]
        except Exception:
            print("Serial read error — reconnecting...")
            self.reconnect()
            return np.zeros(4096, dtype=np.uint32), 0


class D3SController:
    """Threaded controller for the D3S detector."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.detector = None
        self._thread = None
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()

        self.spectrum = np.zeros(4096, dtype=np.uint32)
        self.elapsed_time = 0.0
        self._start_time = None

        # New real-time CPS calculation
        self.cps = 0.0
        self._cps_window = 3.0  # seconds
        self._history = []  # list of (time, total_counts)

        # Logging
        self._log_thread = None
        self._log_stop_event = threading.Event()
        self._log_filename = None
        self._log_interval = None
        self._log_total_time = None
        self.last_delta_t = None

    # --------------------------------------------------------
    # START / STOP
    # --------------------------------------------------------

    def start(self):
        if self._running:
            return
        if self.verbose:
            print("Starting D3SController...")
        try:
            self.detector = D3SDetector()
        except Exception as e:
            print(f"Error connecting to detector: {e}")
            return
        self._stop_event.clear()
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._acquisition_loop)
        self._thread.start()
        if self.verbose:
            print("Acquisition started.")

    def stop(self):
        if not self._running:
            return
        if self.verbose:
            print("Stopping acquisition...")
        self._stop_event.set()
        self.stop_periodic_logging()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._running = False
        if self.detector and hasattr(self.detector, "ser"):
            try:
                if self.detector.ser.is_open:
                    self.detector.ser.close()
            except Exception as e:
                print(f"Serial close error: {e}")
        if self.verbose:
            print("Acquisition stopped cleanly.")

    # --------------------------------------------------------
    # ACQUISITION LOOP
    # --------------------------------------------------------

    def _acquisition_loop(self):
        while not self._stop_event.is_set():
            try:
                spectrum, _ = self.detector.read_spectrum()
                now = time.time()
                with self._lock:
                    self.spectrum += spectrum
                    self.elapsed_time = now - self._start_time

                    # Update CPS history (total counts vs time)
                    total_counts = int(self.spectrum.sum())
                    self._history.append((now, total_counts))

                    # Trim to window length
                    self._history = [(t, c) for t, c in self._history if now - t <= self._cps_window]

                    if len(self._history) > 1:
                        dt = self._history[-1][0] - self._history[0][0]
                        if dt > 0:
                            dc = self._history[-1][1] - self._history[0][1]
                            self.cps = dc / dt
            except Exception as e:
                print(f"Acquisition error: {e}")
                break
            time.sleep(0.05)
        if self.verbose:
            print("Acquisition loop exited cleanly.")

    # --------------------------------------------------------
    # DATA CONTROL
    # --------------------------------------------------------

    def reset(self):
        with self._lock:
            self.spectrum[:] = 0
            self._start_time = time.time()
            self.elapsed_time = 0.0
            self.cps = 0.0
            self._history.clear()
        if self.verbose:
            print("Spectrum reset.")

    def get_spectrum(self):
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

    def acquire_spectrum_for_duration(self, duration, filename=None):
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
    # PERIODIC LOGGING
    # --------------------------------------------------------

    def start_periodic_logging(self, base_filename, interval=10.0, total_time=0.0):
        """Start logging cumulative spectrum every interval seconds."""
        if self._log_thread and self._log_thread.is_alive():
            if self.verbose:
                print("Logging already active.")
            return

        self.reset()  # fresh start
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

        with open(filename, "w") as f:
            f.write("delta_t," + ",".join([f"ch{i}" for i in range(4096)]) + "\n")

        self._log_thread = threading.Thread(target=self._logging_loop)
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
            line = f"{self.last_delta_t:.3f}," + ",".join(map(str, spectrum)) + "\n"
            try:
                with open(self._log_filename, "a") as f:
                    f.write(line)
            except Exception as e:
                print(f"Error writing log file: {e}")

            if self._log_total_time > 0 and (now - start_time) >= self._log_total_time:
                break
            time.sleep(self._log_interval)
        if self.verbose:
            print("Logging loop ended.")

    def stop_periodic_logging(self):
        if self._log_thread and self._log_thread.is_alive():
            self._log_stop_event.set()
            self._log_thread.join(timeout=2.0)
            if self.verbose:
                print("Periodic logging stopped.")
