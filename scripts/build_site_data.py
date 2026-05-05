"""Aggregate the 100 MB microclimate CSV into a small JSON for the website.

Run from the project root:
    python scripts/build_site_data.py

Outputs site_data.json (a few tens of KB) that is embedded inline inside
index.html so the page is fully self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "microclimate-sensors-data.csv"
OUT_PATH = ROOT / "site_data.json"

PARK_COLOR = "#2E8B57"
STREET_COLOR = "#C2513C"
ROOFTOP_COLOR = "#3B6FB6"

SUMMER_MONTHS = [11, 12, 1, 2, 3]
PEAK_MONTHS = [12, 1, 2]
PEAK_HOURS = (13, 17)

SHORT_NAME_MAP = {
    "Birrarung Marr Park - Pole 1131": "Birrarung Marr",
    "Enterprize Park - Pole ID: COM1667": "Enterprize Park",
    "Royal Park Asset ID: COM2707": "Royal Park",
    "Batman Park": "Batman Park",
    "Swanston St - Tram Stop 13 adjacent Federation Sq & Flinders St Station": "Swanston Stop 13",
    "Tram Stop 7B - Melbourne Tennis Centre Precinct - Rod Laver Arena": "Tram Stop 7B",
    "Tram Stop 7C - Melbourne Tennis Centre Precinct - Rod Laver Arena": "Tram Stop 7C",
    "1 Treasury Place": "1 Treasury Pl",
    "101 Collins St L11 Rooftop": "101 Collins L11",
    "CH1 rooftop": "CH1 rooftop",
    "SkyFarm (Jeff's Shed). Rooftop - Melbourne Conference & Exhibition Centre (MCEC)": "MCEC SkyFarm",
}


def classify_site(sensor_location: str) -> str:
    s = sensor_location.lower()
    if any(k in s for k in ["rooftop", "skyfarm", "ch1"]):
        return "rooftop"
    if any(k in s for k in ["birrarung", "enterprize", "royal park", "batman park"]):
        return "park"
    if any(k in s for k in ["tram stop", "swanston", "treasury place"]):
        return "street"
    return "unknown"


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Time"] = pd.to_datetime(df["Time"], utc=True).dt.tz_convert("Australia/Melbourne")
    latlon = df["LatLong"].str.split(",", expand=True)
    df["Latitude"] = pd.to_numeric(latlon[0], errors="coerce")
    df["Longitude"] = pd.to_numeric(latlon[1], errors="coerce")
    df = df.dropna(subset=["SensorLocation", "AirTemperature", "RelativeHumidity"]).copy()
    df["LandType"] = df["SensorLocation"].map(classify_site)
    df = df[df["LandType"] != "unknown"].copy()
    df["SiteShortName"] = df["SensorLocation"].map(SHORT_NAME_MAP).fillna(df["SensorLocation"])
    return df


def hourly_class(df: pd.DataFrame) -> pd.DataFrame:
    site = (
        df.set_index("Time")
        .groupby(["SensorLocation", "LandType"])[["AirTemperature", "RelativeHumidity"]]
        .resample("1h")
        .mean()
        .reset_index()
    )
    return (
        site.groupby(["Time", "LandType"], as_index=False)[["AirTemperature", "RelativeHumidity"]]
        .mean()
    )


def diurnal_profile(hclass: pd.DataFrame, value_col: str, months: list[int]) -> dict:
    sub = hclass[hclass["Time"].dt.month.isin(months)].copy()
    sub["hour"] = sub["Time"].dt.hour
    profile = (
        sub.groupby(["LandType", "hour"], as_index=False)[value_col]
        .mean()
        .pivot(index="hour", columns="LandType", values=value_col)
        .reindex(range(24))
    )
    return {
        "hours": list(range(24)),
        "park": [round(float(v), 3) if pd.notna(v) else None for v in profile.get("park", [])],
        "street": [round(float(v), 3) if pd.notna(v) else None for v in profile.get("street", [])],
        "rooftop": [round(float(v), 3) if pd.notna(v) else None for v in profile.get("rooftop", [])],
    }


def build_gap(hclass: pd.DataFrame) -> pd.DataFrame:
    temp_wide = hclass.pivot(index="Time", columns="LandType", values="AirTemperature")
    rh_wide = hclass.pivot(index="Time", columns="LandType", values="RelativeHumidity")
    g = pd.DataFrame(index=temp_wide.index)
    g["street_minus_park_temp_c"] = temp_wide.get("street") - temp_wide.get("park")
    g["park_minus_street_rh_pct"] = rh_wide.get("park") - rh_wide.get("street")
    g = g.dropna()
    g["hour"] = g.index.hour
    g["date"] = g.index.date
    return g


def main() -> None:
    print(f"Reading {CSV_PATH} ...")
    df = load_clean()
    print(f"  rows kept: {len(df):,}")
    print(f"  date range: {df['Time'].min()} -> {df['Time'].max()}")

    hclass = hourly_class(df)
    gap = build_gap(hclass)

    # Headline stats — summer afternoons (Dec-Feb, 13:00-17:00).
    peak = hclass[
        hclass["Time"].dt.month.isin(PEAK_MONTHS)
        & hclass["Time"].dt.hour.between(*PEAK_HOURS)
    ]
    park_temp = float(peak[peak["LandType"] == "park"]["AirTemperature"].mean())
    street_temp = float(peak[peak["LandType"] == "street"]["AirTemperature"].mean())

    rh_overall = (
        df.groupby("LandType")["RelativeHumidity"].mean().to_dict()
    )

    headline = {
        "park_mean_temp_c": round(park_temp, 2),
        "street_mean_temp_c": round(street_temp, 2),
        "gap_c": round(street_temp - park_temp, 2),
        "park_mean_rh_pct": round(float(rh_overall.get("park", float("nan"))), 1),
        "street_mean_rh_pct": round(float(rh_overall.get("street", float("nan"))), 1),
        "rh_gap_pp": round(
            float(rh_overall.get("park", 0)) - float(rh_overall.get("street", 0)), 1
        ),
        "n_obs": int(len(df)),
        "n_sensors": int(df["SensorLocation"].nunique()),
        "date_start": str(df["Time"].min().date()),
        "date_end": str(df["Time"].max().date()),
    }

    diurnal_temp = diurnal_profile(hclass, "AirTemperature", SUMMER_MONTHS)
    diurnal_rh = diurnal_profile(hclass, "RelativeHumidity", list(range(1, 13)))

    # H4 — hourly cooling gap.
    hourly_gap = (
        gap.groupby("hour", as_index=False)["street_minus_park_temp_c"].mean().sort_values("hour")
    )
    hourly_cooling_gap = {
        "hours": [int(h) for h in hourly_gap["hour"].tolist()],
        "gap_c": [round(float(v), 3) for v in hourly_gap["street_minus_park_temp_c"].tolist()],
    }

    # Day vs night.
    daypart = gap.copy()
    daypart["period"] = np.where(daypart["hour"].between(8, 18), "day", "night")
    daynight = daypart.groupby("period")["street_minus_park_temp_c"].mean().to_dict()
    daynight_gap = {
        "day": round(float(daynight.get("day", 0)), 3),
        "night": round(float(daynight.get("night", 0)), 3),
    }

    # H3 — heatwave amplification.
    daily_max = (
        hclass.assign(date=hclass["Time"].dt.date)
        .groupby("date", as_index=False)["AirTemperature"]
        .max()
        .rename(columns={"AirTemperature": "citywide_max_c"})
    )
    daily_gap = (
        gap[gap["hour"].between(13, 17)]
        .groupby("date", as_index=False)["street_minus_park_temp_c"]
        .mean()
        .rename(columns={"street_minus_park_temp_c": "daytime_gap_c"})
    )
    h3 = daily_gap.merge(daily_max, on="date", how="inner").dropna()
    heatwave_scatter = [
        {
            "date": str(r["date"]),
            "citywide_max_c": round(float(r["citywide_max_c"]), 2),
            "daytime_gap_c": round(float(r["daytime_gap_c"]), 3),
            "is_heatwave": bool(r["citywide_max_c"] > 35),
        }
        for _, r in h3.iterrows()
    ]

    # OLS regression line for the scatter (so the website can draw a clean line).
    if len(h3) >= 2:
        slope, intercept = np.polyfit(h3["citywide_max_c"], h3["daytime_gap_c"], 1)
        x0, x1 = float(h3["citywide_max_c"].min()), float(h3["citywide_max_c"].max())
        regression = {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 3),
            "x0": round(x0, 2),
            "x1": round(x1, 2),
            "y0": round(float(slope * x0 + intercept), 3),
            "y1": round(float(slope * x1 + intercept), 3),
        }
    else:
        regression = None

    # Month x hour heatmap of the cooling signal.
    mh = (
        gap.assign(month=pd.to_datetime(gap["date"]).dt.month)
        .groupby(["month", "hour"])["street_minus_park_temp_c"]
        .mean()
        .unstack("hour")
        .reindex(range(1, 13))
        .reindex(columns=range(24))
    )
    month_hour_heatmap = {
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "hours": list(range(24)),
        "values": [
            [round(float(v), 3) if pd.notna(v) else None for v in row]
            for row in mh.values
        ],
    }

    # Sensors — summer afternoon mean per site.
    peak_site = (
        df[
            df["Time"].dt.month.isin(PEAK_MONTHS)
            & df["Time"].dt.hour.between(*PEAK_HOURS)
        ]
        .groupby(["SensorLocation", "SiteShortName", "LandType"], as_index=False)
        .agg(
            mean_temp_c=("AirTemperature", "mean"),
            latitude=("Latitude", "median"),
            longitude=("Longitude", "median"),
            n_obs=("AirTemperature", "size"),
        )
    )
    sensors = [
        {
            "name": str(r["SiteShortName"]),
            "class": str(r["LandType"]),
            "lat": round(float(r["latitude"]), 5),
            "lon": round(float(r["longitude"]), 5),
            "mean_temp_c": round(float(r["mean_temp_c"]), 2),
            "n_obs": int(r["n_obs"]),
        }
        for _, r in peak_site.iterrows()
    ]

    payload = {
        "headline_stats": headline,
        "palette": {"park": PARK_COLOR, "street": STREET_COLOR, "rooftop": ROOFTOP_COLOR},
        "diurnal_temp": diurnal_temp,
        "diurnal_rh": diurnal_rh,
        "hourly_cooling_gap": hourly_cooling_gap,
        "daynight_gap": daynight_gap,
        "heatwave_scatter": heatwave_scatter,
        "heatwave_regression": regression,
        "month_hour_heatmap": month_hour_heatmap,
        "sensors": sensors,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
