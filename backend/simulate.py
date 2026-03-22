import numpy as np

ZONE_CONFIGS = {
    "Z1": {"base_sst": 27.0, "trend": 0.28, "noise": 0.08, "season_amp": 0.8,  "event": "thermal_stress"},
    "Z2": {"base_sst": 26.5, "trend": 0.18, "noise": 0.07, "season_amp": 0.6,  "event": "hypoxic_bloom"},
    "Z3": {"base_sst": 29.0, "trend": 0.01, "noise": 0.05, "season_amp": 0.4,  "event": "normal"},
    "Z4": {"base_sst": 28.5, "trend": -0.12,"noise": 0.06, "season_amp": 0.5,  "event": "turbidity_spike"},
    "Z5": {"base_sst": 29.5, "trend": -0.22,"noise": 0.06, "season_amp": 0.7,  "event": "upwelling"},
    "Z6": {"base_sst": 26.8, "trend": 0.38, "noise": 0.07, "season_amp": 0.9,  "event": "oil_slick"},
    "Z7": {"base_sst": 28.2, "trend": 0.00, "noise": 0.04, "season_amp": 0.3,  "event": "normal"},
}

def generate_sst_history(zone_id: str, days: int = 90, seed: int = 42) -> np.ndarray:
    """
    Generate realistic SST time-series for a zone.
    Uses a longer warmup period so recent days show clear trend.
    Formula: SST(t) = base + trend*t + season_amp*sin(2π*t/365) + noise
    """
    cfg = ZONE_CONFIGS[zone_id]
    rng = np.random.default_rng(seed)

    # Generate 180 days but only return last `days`
    # This gives the trend time to build up naturally
    total = 180
    t           = np.arange(total)
    trend       = cfg["trend"] * t / 7        # trend per 7-day window
    seasonal    = cfg["season_amp"] * np.sin(2 * np.pi * t / 365)
    noise       = rng.normal(0, cfg["noise"], total)
    sst         = cfg["base_sst"] + trend + seasonal + noise

    return np.round(sst[-days:], 3)

def get_last_n_days(zone_id: str, n: int = 8) -> list:
    """Returns last n days of SST as a plain list."""
    full = generate_sst_history(zone_id, days=90)
    return full[-n:].tolist()

def get_sst_delta(zone_id: str) -> float:
    """SST change over last 8 days."""
    last8 = get_last_n_days(zone_id, 8)
    return round(last8[-1] - last8[0], 3)

def get_slope(zone_id: str) -> float:
    """Linear slope of SST over last 8 days (°C per day)."""
    last8 = np.array(get_last_n_days(zone_id, 8))
    t     = np.arange(len(last8))
    slope = np.polyfit(t, last8, 1)[0]
    return round(float(slope), 4)

def get_all_zone_sst() -> dict:
    """Returns SST history + stats for all zones."""
    result = {}
    for zid in ZONE_CONFIGS:
        result[zid] = {
            "history": get_last_n_days(zid, 8),
            "delta":   get_sst_delta(zid),
            "slope":   get_slope(zid),
        }
    return result

if __name__ == "__main__":
    data = get_all_zone_sst()
    for zid, vals in data.items():
        print(f"{zid}: delta={vals['delta']:+.2f}°C  slope={vals['slope']:+.4f}  last={vals['history'][-1]:.2f}°C")