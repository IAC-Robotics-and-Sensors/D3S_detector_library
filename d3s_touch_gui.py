"""
D3S Touchscreen GUI
===================

A touch-friendly D3S GUI with larger controls for small displays.
Designed for Linux touchscreens (for example 5-7 inch panels).
"""

from __future__ import annotations

import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from d3s_controller import D3SController


TOUCH_STYLE = """
QMainWindow {
    background: #111111;
}
QLabel {
    color: #f2f2f2;
    font-size: 18px;
}
QGroupBox {
    color: #f2f2f2;
    border: 2px solid #3a3a3a;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 10px;
    font-size: 18px;
    font-weight: 600;
}
QPushButton {
    min-height: 64px;
    min-width: 120px;
    font-size: 20px;
    font-weight: 700;
    border-radius: 12px;
    border: 1px solid #444444;
    background: #2d2d2d;
    color: #ffffff;
    padding: 8px;
}
QPushButton:pressed {
    background: #4a4a4a;
}
QPushButton:disabled {
    background: #202020;
    color: #8a8a8a;
}
QTextEdit {
    background: #1a1a1a;
    color: #f2f2f2;
    font-size: 16px;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
}
QSpinBox {
    min-height: 44px;
    min-width: 120px;
    font-size: 18px;
    background: #1f1f1f;
    color: #ffffff;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 4px;
}
QCheckBox {
    color: #f2f2f2;
    font-size: 18px;
    spacing: 10px;
}
QTabWidget::pane {
    border: 1px solid #3a3a3a;
}
QTabBar::tab {
    min-height: 38px;
    min-width: 120px;
    font-size: 16px;
    background: #1f1f1f;
    color: #e8e8e8;
    padding: 6px 10px;
}
QTabBar::tab:selected {
    background: #2f2f2f;
}
"""


class MainWindow(QMainWindow):
    """Touchscreen-oriented GUI for the Kromek D3S detector."""

    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("D3S Touchscreen GUI")
        self.resize(1024, 600)

        self.controller = D3SController(verbose=True)
        self._acquiring = False

        self._build_ui()
        self._connect_signals()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)

        self.log_signal.connect(self._append_log)

        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        tabs = QTabWidget()
        root.addWidget(tabs)

        live_tab = QWidget()
        tabs.addTab(live_tab, "Live")
        self._build_live_tab(live_tab)

        acq_tab = QWidget()
        tabs.addTab(acq_tab, "Acquisition")
        self._build_acq_tab(acq_tab)

        device_tab = QWidget()
        tabs.addTab(device_tab, "Device")
        self._build_device_tab(device_tab)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        root.addWidget(self.log_box)

        self.setStatusBar(QStatusBar())

    def _build_live_tab(self, parent: QWidget):
        lay = QVBoxLayout(parent)

        self.fig = Figure(figsize=(8, 3.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Channel")
        self.ax.set_ylabel("Counts")
        self.ax.set_title("D3S Live Spectrum")
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas)
        self._line = None

        button_group = QGroupBox("Main Controls")
        grid = QGridLayout(button_group)
        self.btn_connect = QPushButton("Connect")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_clear = QPushButton("Clear")
        self.btn_save = QPushButton("Save As")
        self.btn_quick_save = QPushButton("Quick Save")
        self.chk_log_scale = QCheckBox("Log scale")

        self.btn_quick_10 = QPushButton("Acquire 10s")
        self.btn_quick_30 = QPushButton("Acquire 30s")
        self.btn_quick_60 = QPushButton("Acquire 60s")

        grid.addWidget(self.btn_connect, 0, 0)
        grid.addWidget(self.btn_disconnect, 0, 1)
        grid.addWidget(self.btn_start, 0, 2)
        grid.addWidget(self.btn_stop, 0, 3)
        grid.addWidget(self.btn_clear, 1, 0)
        grid.addWidget(self.btn_save, 1, 1)
        grid.addWidget(self.btn_quick_save, 1, 2)
        grid.addWidget(self.chk_log_scale, 1, 3)
        grid.addWidget(self.btn_quick_10, 2, 0)
        grid.addWidget(self.btn_quick_30, 2, 1)
        grid.addWidget(self.btn_quick_60, 2, 2)

        lay.addWidget(button_group)

        status_group = QGroupBox("Status")
        status_layout = QHBoxLayout(status_group)
        self.lbl_status = QLabel("Status: Disconnected")
        self.lbl_counts = QLabel("Counts: 0")
        self.lbl_cps = QLabel("CPS: 0.0")
        self.lbl_neutrons = QLabel("Neutrons: 0")
        self.lbl_elapsed = QLabel("Time: 0.0 s")

        status_layout.addWidget(self.lbl_status)
        status_layout.addStretch()
        status_layout.addWidget(self.lbl_counts)
        status_layout.addWidget(self.lbl_cps)
        status_layout.addWidget(self.lbl_neutrons)
        status_layout.addWidget(self.lbl_elapsed)
        lay.addWidget(status_group)

    def _build_acq_tab(self, parent: QWidget):
        lay = QVBoxLayout(parent)

        timed_group = QGroupBox("Timed Acquisition")
        row1 = QHBoxLayout(timed_group)
        row1.addWidget(QLabel("Duration (s):"))
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(1, 100000)
        self.spin_duration.setValue(30)
        self.spin_duration.setSuffix(" s")
        row1.addWidget(self.spin_duration)
        self.btn_acquire_timed = QPushButton("Acquire Timed")
        row1.addWidget(self.btn_acquire_timed)
        row1.addStretch()
        lay.addWidget(timed_group)

        logging_group = QGroupBox("Periodic Logging")
        row2 = QHBoxLayout(logging_group)
        row2.addWidget(QLabel("Interval (s):"))
        self.spin_log_interval = QSpinBox()
        self.spin_log_interval.setRange(1, 100000)
        self.spin_log_interval.setValue(10)
        self.spin_log_interval.setSuffix(" s")
        row2.addWidget(self.spin_log_interval)

        row2.addWidget(QLabel("Total (s, 0=continuous):"))
        self.spin_log_total = QSpinBox()
        self.spin_log_total.setRange(0, 1000000)
        self.spin_log_total.setValue(0)
        self.spin_log_total.setSuffix(" s")
        row2.addWidget(self.spin_log_total)

        self.btn_start_logging = QPushButton("Start Logging")
        self.btn_stop_logging = QPushButton("Stop Logging")
        row2.addWidget(self.btn_start_logging)
        row2.addWidget(self.btn_stop_logging)
        row2.addStretch()
        lay.addWidget(logging_group)

        lay.addStretch()

    def _build_device_tab(self, parent: QWidget):
        lay = QVBoxLayout(parent)

        info_group = QGroupBox("Device Info")
        info_layout = QVBoxLayout(info_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Serial Number:"))
        self.lbl_serial = QLabel("-")
        row1.addWidget(self.lbl_serial)
        row1.addStretch()
        info_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Temperature:"))
        self.lbl_temp = QLabel("- degC")
        row2.addWidget(self.lbl_temp)
        self.btn_read_temp = QPushButton("Read Temperature")
        row2.addWidget(self.btn_read_temp)
        row2.addStretch()
        info_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_fullscreen = QPushButton("Toggle Fullscreen")
        row3.addWidget(self.btn_fullscreen)
        row3.addStretch()
        info_layout.addLayout(row3)

        lay.addWidget(info_group)
        lay.addStretch()

    def _connect_signals(self):
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_quick_save.clicked.connect(self._on_quick_save)

        self.btn_quick_10.clicked.connect(lambda: self._quick_acquire(10))
        self.btn_quick_30.clicked.connect(lambda: self._quick_acquire(30))
        self.btn_quick_60.clicked.connect(lambda: self._quick_acquire(60))

        self.chk_log_scale.stateChanged.connect(self._on_log_scale)

        self.btn_acquire_timed.clicked.connect(self._on_acquire_timed)
        self.btn_start_logging.clicked.connect(self._on_start_logging)
        self.btn_stop_logging.clicked.connect(self._on_stop_logging)

        self.btn_read_temp.clicked.connect(self._on_read_temp)
        self.btn_fullscreen.clicked.connect(self._on_toggle_fullscreen)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_controls_enabled(self, on: bool):
        for w in (
            self.btn_start,
            self.btn_stop,
            self.btn_clear,
            self.btn_save,
            self.btn_quick_save,
            self.btn_quick_10,
            self.btn_quick_30,
            self.btn_quick_60,
            self.btn_acquire_timed,
            self.btn_start_logging,
            self.btn_stop_logging,
            self.btn_read_temp,
        ):
            w.setEnabled(on)

    def _log(self, msg: str):
        self.log_signal.emit(msg)

    @pyqtSlot(str)
    def _append_log(self, msg: str):
        self.log_box.append(msg)

    def _acquire_for_duration(self, duration: int, save_path: str | None = None):
        self._log(f"Acquiring for {duration}s...")

        def worker():
            try:
                _, elapsed = self.controller.acquire_spectrum_for_duration(duration, save_path)
                if save_path:
                    self._log(f"Timed acquisition complete ({elapsed:.1f}s). Saved to {save_path}")
                else:
                    self._log(f"Timed acquisition complete ({elapsed:.1f}s).")
            except Exception as exc:
                self._log(f"Timed acquisition failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_connect(self):
        self._log("Connecting to D3S...")
        if self.controller.connect():
            sn = self.controller.serial_number
            self.lbl_status.setText(f"Status: Connected ({sn})")
            self.lbl_serial.setText(sn or "-")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self._set_controls_enabled(True)
            self.btn_stop.setEnabled(False)
            self._log(f"Connected (S/N: {sn})")
        else:
            QMessageBox.warning(self, "Connection", "D3S device not found. Check USB connection.")

    def _on_disconnect(self):
        if self._acquiring:
            self._on_stop()
        self.controller.disconnect()
        self.lbl_status.setText("Status: Disconnected")
        self.lbl_serial.setText("-")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self._set_controls_enabled(False)
        self._log("Disconnected.")

    def _on_start(self):
        if not self.controller.is_connected:
            return
        self.controller.start()
        self._acquiring = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._timer.start(100)
        self._log("Acquisition started.")

    def _on_stop(self):
        self.controller.stop()
        self._acquiring = False
        self._timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._log("Acquisition stopped.")

    def _on_clear(self):
        self.controller.reset()
        self._log("Spectrum cleared.")

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Spectrum", "", "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        spec, elapsed, _ = self.controller.get_spectrum()
        np.savetxt(path, spec)
        self._log(f"Spectrum ({elapsed:.1f}s) saved to {path}")

    def _on_quick_save(self):
        spec, elapsed, _ = self.controller.get_spectrum()
        out_dir = Path.cwd()
        out_path = out_dir / f"d3s_touch_spectrum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        np.savetxt(out_path, spec)
        self._log(f"Quick saved spectrum ({elapsed:.1f}s) to {out_path}")

    def _quick_acquire(self, duration: int):
        self.controller.reset()
        self._acquire_for_duration(duration)

    def _on_log_scale(self, state):
        self.ax.set_yscale("log" if state else "linear")
        self.canvas.draw_idle()

    def _on_acquire_timed(self):
        duration = self.spin_duration.value()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Timed Spectrum", "", "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        self.controller.reset()
        self._acquire_for_duration(duration, path)

    def _on_start_logging(self):
        interval = self.spin_log_interval.value()
        total = self.spin_log_total.value()
        path, _ = QFileDialog.getSaveFileName(
            self, "Log Base Filename", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self.controller.reset()
        self.controller.start_periodic_logging(path, interval, total)
        self._log(f"Logging started ({interval}s interval).")

    def _on_stop_logging(self):
        self.controller.stop_periodic_logging()
        self._log("Logging stopped.")

    def _on_read_temp(self):
        temp = self.controller.read_temperature()
        if np.isnan(temp):
            self._log("Failed to read temperature.")
            return
        self.lbl_temp.setText(f"{temp:.0f} degC")
        self._log(f"Temperature = {temp:.0f} degC")

    def _on_toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self._log("Exited fullscreen.")
        else:
            self.showFullScreen()
            self._log("Entered fullscreen.")

    def _refresh(self):
        if not self._acquiring:
            return

        spec, elapsed, cps = self.controller.get_spectrum()
        total_counts = int(spec.sum())
        neutrons = self.controller.neutron_count

        self.lbl_counts.setText(f"Counts: {total_counts}")
        self.lbl_cps.setText(f"CPS: {cps:.1f}")
        self.lbl_neutrons.setText(f"Neutrons: {neutrons}")
        self.lbl_elapsed.setText(f"Time: {elapsed:.1f} s")

        delta_t = self.controller.last_delta_t
        if delta_t is not None:
            self.statusBar().showMessage(f"Elapsed: {elapsed:.1f}s | CPS: {cps:.1f} | dt: {delta_t:.2f}s")
        else:
            self.statusBar().showMessage(f"Elapsed: {elapsed:.1f}s | CPS: {cps:.1f}")

        x = np.arange(len(spec), dtype=float)

        if self._line is None:
            self._line, = self.ax.plot(x, spec, linewidth=1.0)
            self.ax.set_xlim(x.min(), x.max())
        else:
            self._line.set_xdata(x)
            self._line.set_ydata(spec)
            self.ax.set_xlim(x.min(), x.max())

        ymax = max(spec.max(), 1)
        if self.chk_log_scale.isChecked():
            self.ax.set_ylim(0.5, ymax * 2)
        else:
            self.ax.set_ylim(0, ymax * 1.1)

        self.canvas.draw_idle()

    def closeEvent(self, event):
        if self._acquiring:
            self._on_stop()
        self.controller.disconnect()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(TOUCH_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
