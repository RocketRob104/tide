"""
Offline-friendly tide time estimator with a small Tkinter GUI.

This program uses a deliberately simple harmonic tide model so that it can
operate with no network access or external datasets. It is educational and
approximate: real-world tides depend heavily on local geography, friction,
and dozens of harmonic constituents measured at tide stations. Here we model
only a handful of large constituents (M2, S2, K1, O1) and scale them with
latitude/longitude heuristics. The output is useful for demonstrations and
rough intuition but should not be used for navigation or safety-critical work.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from tkinter import messagebox, ttk
from typing import Iterable, List, Tuple

# --- Core tidal math ------------------------------------------------------
# The goal of this section is to provide a self-contained way to estimate
# tidal heights and find nearby high/low tide times without any downloaded
# tables. Accuracy is intentionally modest but the structure mirrors how
# professional models combine harmonic constituents.


@dataclass
class HarmonicConstituent:
    """Represents a single tidal constituent (one repeating wave component)."""

    name: str
    period_hours: float  # Time for one full cycle.
    amplitude_m: float  # Peak contribution in meters at the equator.
    phase_radians: float  # Phase offset to desynchronize the waves slightly.


@dataclass
class TideEvent:
    """A detected high or low tide."""

    time: datetime
    height_m: float
    label: str  # "High" or "Low"


# Epoch is used as a stable reference so the harmonic phases stay consistent.
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

# Four strong constituents. Amplitudes are normalized for open-ocean behavior;
# the heuristic scaling function will adjust them for latitude.
CONSTITUENTS: Tuple[HarmonicConstituent, ...] = (
    HarmonicConstituent("M2", period_hours=12.42, amplitude_m=1.0, phase_radians=0.3),
    HarmonicConstituent("S2", period_hours=12.00, amplitude_m=0.46, phase_radians=-0.2),
    HarmonicConstituent("K1", period_hours=23.93, amplitude_m=0.30, phase_radians=0.6),
    HarmonicConstituent("O1", period_hours=25.82, amplitude_m=0.20, phase_radians=-0.4),
)


def hours_since_epoch(moment: datetime) -> float:
    """Return hours from the fixed epoch, keeping everything in UTC."""

    utc_time = moment.astimezone(timezone.utc)
    delta = utc_time - EPOCH
    return delta.total_seconds() / 3600.0


def longitude_phase_adjustment(longitude_deg: float) -> float:
    """
    Convert longitude into an offset in hours.

    360 degrees corresponds to one full daily rotation (24 hours), so each
    degree of longitude advances local solar time by ~4 minutes. This lets the
    model align peaks roughly with local solar/lunar time instead of UTC.
    """

    return longitude_deg / 15.0  # 360 deg / 24 h = 15 deg per hour.


def latitude_amplitude_scale(latitude_deg: float) -> float:
    """
    Scale constituent amplitudes with latitude.

    Real tides are stronger in some latitudes due to resonance and Earth's
    tilt. We gently reduce amplitude toward the poles and never let it reach
    zero. The returned multiplier stays within [0.35, 1.0].
    """

    lat_rad = math.radians(latitude_deg)
    return max(0.35, abs(math.cos(lat_rad)))


def tide_height(moment: datetime, latitude_deg: float, longitude_deg: float) -> float:
    """
    Compute an approximate tidal height in meters at a given time and location.

    The height is the sum of cosine waves (our constituents). Each constituent
    has:
    - A period describing how quickly it repeats.
    - An amplitude describing how much it contributes.
    - A phase offset to keep the waves from aligning perfectly.

    We also inject a longitude-driven offset so the waves crest earlier or
    later depending on where you are on Earth.
    """

    t_hours = hours_since_epoch(moment)
    solar_phase_hours = longitude_phase_adjustment(longitude_deg)
    lat_scale = latitude_amplitude_scale(latitude_deg)

    height = 0.0
    for c in CONSTITUENTS:
        # Convert the running clock into a position within the constituent's
        # current cycle, then turn that into an angle for the cosine function.
        phase_in_cycle = (t_hours + solar_phase_hours) / c.period_hours
        angle = 2.0 * math.pi * phase_in_cycle + c.phase_radians
        height += lat_scale * c.amplitude_m * math.cos(angle)

    return height


def refine_extremum(
    prev_point: Tuple[datetime, float],
    mid_point: Tuple[datetime, float],
    next_point: Tuple[datetime, float],
) -> Tuple[datetime, float]:
    """
    Improve the time/height estimate for a detected peak using a parabola.

    We fit a quadratic through three consecutive points and compute where its
    derivative crosses zero. This reduces the timing error introduced by our
    coarse sampling interval.
    """

    # Use seconds relative to the middle point to keep the math small.
    step_seconds = (next_point[0] - mid_point[0]).total_seconds()
    if step_seconds == 0:
        return mid_point

    y_prev, y_mid, y_next = prev_point[1], mid_point[1], next_point[1]

    # Quadratic coefficients for equally spaced samples.
    a = (y_next + y_prev - 2 * y_mid) / (2 * step_seconds ** 2)
    b = (y_next - y_prev) / (2 * step_seconds)

    if a == 0:
        return mid_point

    # t_extreme is relative to the middle sample.
    t_extreme = -b / (2 * a)
    refined_time = mid_point[0] + timedelta(seconds=t_extreme)
    refined_height = y_mid + a * t_extreme ** 2 + b * t_extreme
    return refined_time, refined_height


def detect_tide_events(
    samples: Iterable[Tuple[datetime, float]]
) -> List[TideEvent]:
    """
    Scan a list of (time, height) samples and pick out highs and lows.

    We look at every trio of points and flag local maxima/minima. Then we
    refine each extremum with a quadratic fit.
    """

    points = list(samples)
    events: List[TideEvent] = []

    for i in range(1, len(points) - 1):
        prev_point, mid_point, next_point = points[i - 1], points[i], points[i + 1]
        if mid_point[1] > prev_point[1] and mid_point[1] > next_point[1]:
            refined_time, refined_height = refine_extremum(prev_point, mid_point, next_point)
            events.append(TideEvent(refined_time, refined_height, "High"))
        elif mid_point[1] < prev_point[1] and mid_point[1] < next_point[1]:
            refined_time, refined_height = refine_extremum(prev_point, mid_point, next_point)
            events.append(TideEvent(refined_time, refined_height, "Low"))

    # Sort chronologically in case refinement nudged times out of order.
    events.sort(key=lambda e: e.time)
    return events


def predict_tides_for_day(
    target_date: date, latitude_deg: float, longitude_deg: float, tzinfo: timezone, step_minutes: int = 10
) -> List[TideEvent]:
    """
    Predict high/low tides for a single day.

    We sample the 24-hour window (plus one step on either side so that peaks
    near midnight are still found) and then run peak detection.
    """

    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive.")

    # Start one step before midnight and end one step after to catch edges.
    start = datetime.combine(target_date, datetime.min.time(), tzinfo=tzinfo) - timedelta(minutes=step_minutes)
    end = start + timedelta(days=1, minutes=2 * step_minutes)
    step = timedelta(minutes=step_minutes)

    samples: List[Tuple[datetime, float]] = []
    current = start
    while current <= end:
        height = tide_height(current, latitude_deg, longitude_deg)
        samples.append((current, height))
        current += step

    events = detect_tide_events(samples)

    # Keep only the events that fall inside the target day (midnight to midnight).
    day_start = start + timedelta(minutes=step_minutes)
    day_end = day_start + timedelta(days=1)
    return [e for e in events if day_start <= e.time <= day_end]


# --- GUI ------------------------------------------------------------------
# The GUI is intentionally lightweight so it can run anywhere Tkinter is
# available. Everything lives in one file for simplicity.


class TideApp(tk.Tk):
    """Small Tkinter app that wraps the tidal predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Offline Tide Time Estimator")
        self.geometry("740x520")
        self.resizable(False, False)

        self._build_form()
        self._build_results_table()
        self._build_notes()

    # UI construction helpers ------------------------------------------------
    def _build_form(self) -> None:
        """Create user inputs for latitude, longitude, and date."""

        frame = ttk.LabelFrame(self, text="Location and Date")
        frame.pack(fill="x", padx=12, pady=10)

        ttk.Label(frame, text="Latitude (deg, south is negative):").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.lat_var = tk.StringVar(value="37.7749")
        ttk.Entry(frame, textvariable=self.lat_var, width=20).grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frame, text="Longitude (deg, west is negative):").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.lon_var = tk.StringVar(value="-122.4194")
        ttk.Entry(frame, textvariable=self.lon_var, width=20).grid(row=1, column=1, padx=6, pady=4, sticky="w")

        today_str = datetime.now().date().isoformat()
        ttk.Label(frame, text="Date (YYYY-MM-DD):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.date_var = tk.StringVar(value=today_str)
        ttk.Entry(frame, textvariable=self.date_var, width=20).grid(row=2, column=1, padx=6, pady=4, sticky="w")

        self.tzinfo = datetime.now().astimezone().tzinfo or timezone.utc
        tz_label = f"Using local timezone: {self.tzinfo}"
        ttk.Label(frame, text=tz_label).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        ttk.Button(frame, text="Predict Tides", command=self.on_predict).grid(row=4, column=0, columnspan=2, pady=8)

    def _build_results_table(self) -> None:
        """Create a Treeview to display the predicted tides."""

        frame = ttk.LabelFrame(self, text="Predicted High / Low Tides")
        frame.pack(fill="both", expand=True, padx=12, pady=6)

        columns = ("time", "type", "height")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        self.tree.heading("time", text="Local Time")
        self.tree.heading("type", text="High/Low")
        self.tree.heading("height", text="Height (m)")
        self.tree.column("time", width=220)
        self.tree.column("type", width=80, anchor="center")
        self.tree.column("height", width=100, anchor="e")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_notes(self) -> None:
        """Add a disclaimer and short usage instructions."""

        frame = ttk.LabelFrame(self, text="Notes")
        frame.pack(fill="x", padx=12, pady=6)

        notes = (
            "• Offline and educational: tides vary dramatically with local geography.\n"
            "• Model uses four major harmonic constituents (M2, S2, K1, O1).\n"
            "• Heights are relative, not tied to a datum; treat as approximate meters.\n"
            "• Reduce the sampling interval in code for finer timing (uses more CPU)."
        )
        label = ttk.Label(frame, text=notes, justify="left")
        label.pack(fill="x", padx=6, pady=4)

    # Event handlers --------------------------------------------------------
    def on_predict(self) -> None:
        """Parse inputs, run the predictor, and render the results."""

        try:
            latitude = float(self.lat_var.get())
            longitude = float(self.lon_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Latitude and longitude must be numbers.")
            return

        try:
            target_date = datetime.strptime(self.date_var.get(), "%Y-%m-%d").date()
        except ValueError:
            messagebox.showerror("Invalid date", "Use the YYYY-MM-DD format for the date.")
            return

        events = predict_tides_for_day(target_date, latitude, longitude, self.tzinfo)

        # Clear previous rows.
        for row in self.tree.get_children():
            self.tree.delete(row)

        for event in events:
            time_str = event.time.astimezone(self.tzinfo).strftime("%Y-%m-%d %H:%M")
            height_str = f"{event.height_m:+.2f}"
            self.tree.insert("", "end", values=(time_str, event.label, height_str))

        if not events:
            messagebox.showinfo("No tides detected", "No high or low tides found for that day. "
                                                     "Try widening the date window or reducing the step size.")


def main() -> None:
    """Entry point for running the Tkinter application."""

    app = TideApp()
    app.mainloop()


if __name__ == "__main__":
    main()
