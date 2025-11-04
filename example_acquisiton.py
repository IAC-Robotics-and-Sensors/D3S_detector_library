"""
Example: Using D3SController for Timed and Periodic Acquisitions
================================================================

This script demonstrates how to use the D3SController class to:
 1. Acquire a fixed-duration cumulative spectrum and plot it.
 2. Perform a periodic acquisition, logging cumulative spectra to a CSV file
    while showing a live updating plot.

Requirements:
    pip install pyserial numpy matplotlib

Usage:
    python example_acquisition.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from d3s_controller import D3SController


def example_timed_acquisition():
    """
    Example 1: Acquire a single spectrum for a fixed duration and plot it.
    """
    print("\n=== Timed Acquisition Example ===")
    ctrl = D3SController(verbose=True)
    ctrl.start()
    time.sleep(0.5)  # Give time to start
    ctrl.reset()

    duration = 10.0  # seconds
    elapsed_time = 0.0
    spectrum = np.zeros(4096, dtype=np.uint32)
    filename = "example_timed_spectrum.txt"

    print(f"Acquiring spectrum for {duration:.1f} seconds...")
    
    # Live updating plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    line, =ax.plot(spectrum, color="blue")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")
    plt.tight_layout()
    while elapsed_time < duration:
        spectrum, elapsed_time, cps = ctrl.get_spectrum()
        print(f"Elapsed: {elapsed_time:.1f}s | CPS: {cps:.1f}")

        # Update plot
        line.set_ydata(spectrum)
        ax.relim()
        ax.autoscale_view(True, True, True)
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    np.savetxt(filename, spectrum)

    print(f"\nAcquisition complete.")
    print(f"Elapsed time: {elapsed_time:.2f} s")
    print(f"Spectrum saved to: {filename}")
    print(f"Total counts: {int(np.sum(spectrum))}\n")

    ctrl.stop()
    time.sleep(1)


def example_periodic_logging():
    """
    Example 2: Continuously log spectra to a CSV file every few seconds,
    updating a live plot as counts accumulate.
    """
    print("\n=== Periodic Logging Example ===")
    ctrl = D3SController(verbose=True)
    ctrl.start()

    interval = 15.0  # seconds between saved spectra
    total_time = 60.0  # run for 1 minute (shorter for demo)
    base_filename = "example_periodic_log"

    print(f"Starting periodic logging every {interval}s for {total_time}s...")
    ctrl.start_periodic_logging(base_filename, interval, total_time)

    # Live updating plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot(np.zeros(4096), color="green")
    ax.set_title("Periodic Logging (Cumulative Spectrum)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")
    ax.grid(True)
    plt.tight_layout()

    start = time.time()
    try:
        while ctrl._log_thread and ctrl._log_thread.is_alive():
            spectrum, elapsed, cps = ctrl.get_spectrum()
            line.set_ydata(spectrum)
            ax.relim()
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            fig.canvas.flush_events()

            print(f"Elapsed: {elapsed:.1f}s | CPS: {cps:.1f} | Δt: {ctrl.last_delta_t or 0:.2f}s")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Interrupted — stopping logging.")
        ctrl.stop_periodic_logging()

    ctrl.stop()
    plt.ioff()
    print(f"\nLogging complete. File written to {ctrl._log_filename}\n")


if __name__ == "__main__":
    print("D3S Detector Example Program")
    print("============================")
    print("1. Timed Acquisition (10 s)")
    print("2. Periodic Logging (15 s interval for 1 min)\n")

    try:
        example_timed_acquisition()
        example_periodic_logging()
    except KeyboardInterrupt:
        print("\nUser interrupted. Stopping acquisition.")
