import numpy as np
from dtaidistance import dtw
from simulate import generate_sst_history, get_slope, ZONE_CONFIGS

# ─────────────────────────────────────────────
# SIGNATURE TEMPLATES
# Each signature = ordered list of steps.
# Each step = { variable, direction, window_days }
# This is what makes AIRAVAT different from threshold alerts.
# ─────────────────────────────────────────────

SIGNATURES = {
    "thermal_stress": {
        "steps": [
            {"step": 1, "name": "Baseline normal",         "variable": "sst", "direction": "stable",   "threshold": 0.05},
            {"step": 2, "name": "SST rise begins",         "variable": "sst", "direction": "rising",   "threshold": 0.10},
            {"step": 3, "name": "Sustained warming",       "variable": "sst", "direction": "rising",   "threshold": 0.20},
            {"step": 4, "name": "Acceleration detected",   "variable": "sst", "direction": "rising",   "threshold": 0.28},
            {"step": 5, "name": "Thermal anomaly confirmed","variable": "sst","direction": "rising",   "threshold": 0.32},
            {"step": 6, "name": "Crisis threshold near",   "variable": "sst", "direction": "rising",   "threshold": 0.36},
            {"step": 7, "name": "Critical — event imminent","variable": "sst","direction": "rising",   "threshold": 0.40},
        ],
        # Reference SST pattern for DTW matching (normalised shape)
        "reference": [0.0, 0.1, 0.25, 0.45, 0.65, 0.82, 1.0]
    },

    "hypoxic_bloom": {
        "steps": [
            {"step": 1, "name": "Baseline normal",         "variable": "sst", "direction": "stable",   "threshold": 0.05},
            {"step": 2, "name": "Nutrient loading starts", "variable": "sst", "direction": "rising",   "threshold": 0.08},
            {"step": 3, "name": "DO drop observed",        "variable": "sst", "direction": "rising",   "threshold": 0.14},
            {"step": 4, "name": "Chl-a spike detected",    "variable": "sst", "direction": "rising",   "threshold": 0.18},
            {"step": 5, "name": "Bloom initiation",        "variable": "sst", "direction": "rising",   "threshold": 0.22},
            {"step": 6, "name": "Hypoxia spreading",       "variable": "sst", "direction": "rising",   "threshold": 0.26},
            {"step": 7, "name": "Critical hypoxia",        "variable": "sst", "direction": "rising",   "threshold": 0.30},
        ],
        "reference": [0.0, 0.08, 0.18, 0.30, 0.45, 0.62, 0.80]
    },

    "turbidity_spike": {
        "steps": [
            {"step": 1, "name": "Baseline normal",         "variable": "sst", "direction": "stable",   "threshold": 0.05},
            {"step": 2, "name": "Sediment load rising",    "variable": "sst", "direction": "falling",  "threshold": -0.05},
            {"step": 3, "name": "Visibility drop",         "variable": "sst", "direction": "falling",  "threshold": -0.10},
            {"step": 4, "name": "High turbidity zone",     "variable": "sst", "direction": "falling",  "threshold": -0.15},
            {"step": 5, "name": "Event confirmed",         "variable": "sst", "direction": "falling",  "threshold": -0.20},
        ],
        "reference": [0.0, -0.05, -0.12, -0.20, -0.30]
    },

    "upwelling": {
        "steps": [
            {"step": 1, "name": "Baseline normal",         "variable": "sst", "direction": "stable",   "threshold": 0.05},
            {"step": 2, "name": "Cold upwelling starts",   "variable": "sst", "direction": "falling",  "threshold": -0.10},
            {"step": 3, "name": "SST drop confirmed",      "variable": "sst", "direction": "falling",  "threshold": -0.20},
            {"step": 4, "name": "Nutrient rich zone",      "variable": "sst", "direction": "falling",  "threshold": -0.30},
            {"step": 5, "name": "Fishery opportunity",     "variable": "sst", "direction": "falling",  "threshold": -0.40},
            {"step": 6, "name": "Peak upwelling",          "variable": "sst", "direction": "falling",  "threshold": -0.50},
            {"step": 7, "name": "Upwelling subsiding",     "variable": "sst", "direction": "stable",   "threshold": 0.05},
        ],
        "reference": [0.0, -0.12, -0.28, -0.45, -0.60, -0.72, -0.65]
    },

    "oil_slick": {
        "steps": [
            {"step": 1, "name": "Baseline normal",         "variable": "sst", "direction": "stable",   "threshold": 0.05},
            {"step": 2, "name": "SAR anomaly detected",    "variable": "sst", "direction": "rising",   "threshold": 0.12},
            {"step": 3, "name": "Surface film forming",    "variable": "sst", "direction": "rising",   "threshold": 0.22},
            {"step": 4, "name": "Spreading confirmed",     "variable": "sst", "direction": "rising",   "threshold": 0.32},
            {"step": 5, "name": "Slick boundary mapped",   "variable": "sst", "direction": "rising",   "threshold": 0.42},
            {"step": 6, "name": "Source identified",       "variable": "sst", "direction": "rising",   "threshold": 0.52},
            {"step": 7, "name": "Containment needed",      "variable": "sst", "direction": "rising",   "threshold": 0.62},
        ],
        "reference": [0.0, 0.12, 0.28, 0.46, 0.60, 0.75, 0.90]
    },

    "normal": {
        "steps": [
            {"step": 1, "name": "Baseline normal",         "variable": "sst", "direction": "stable",   "threshold": 0.05},
        ],
        "reference": [0.0]
    }
}

# ─────────────────────────────────────────────
# DTW MATCHER
# Compares live SST (normalised) against each
# signature reference using dynamic time warping.
# Returns best match + confidence score.
# ─────────────────────────────────────────────

def normalise(series: np.ndarray) -> np.ndarray:
    """Min-max normalise a series to [0, 1] range."""
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-6:
        return np.zeros_like(series)
    return (series - mn) / (mx - mn)

def dtw_match(zone_id: str) -> dict:
    """
    Run DTW against all signature templates.
    Returns the best matching signature + step index + confidence.
    """
    sst_full  = generate_sst_history(zone_id, days=90)
    sst_last8 = sst_full[-8:]
    sst_norm  = normalise(sst_last8)

    best_sig   = "normal"
    best_dist  = float("inf")
    best_conf  = 0.0

    for sig_name, sig_data in SIGNATURES.items():
        ref      = np.array(sig_data["reference"], dtype=float)
        # DTW distance between live zone and reference shape
        dist     = dtw.distance_fast(sst_norm.astype(np.double),
                                     ref.astype(np.double))
        if dist < best_dist:
            best_dist = dist
            best_sig  = sig_name

    # Confidence: inverse of distance, scaled to 0-1
    # distance of 0 = perfect match = 1.0 confidence
    # distance of 2+ = poor match = ~0.0 confidence
    best_conf = float(np.clip(1.0 - (best_dist / 2.0), 0.0, 1.0))

    return {
        "zone_id":    zone_id,
        "signature":  best_sig,
        "dtw_dist":   round(best_dist, 4),
        "dtw_conf":   round(best_conf, 4),
    }

# ─────────────────────────────────────────────
# STEP DETECTOR
# Given the DTW match, figure out which step
# in the chain the zone is currently at.
# ─────────────────────────────────────────────

def detect_step(zone_id: str, signature: str) -> int:
    """
    Walk the signature steps in order.
    Return the highest step whose threshold is satisfied.
    """
    sst_full  = generate_sst_history(zone_id, days=90)
    sst_last8 = sst_full[-8:]
    slope     = get_slope(zone_id)
    delta     = float(sst_last8[-1] - sst_last8[0])

    steps     = SIGNATURES[signature]["steps"]
    confirmed = 1   # always at least step 1

    for step_def in steps:
        thresh = step_def["threshold"]
        direct = step_def["direction"]

        if direct == "rising"  and slope >= thresh: confirmed = step_def["step"]
        if direct == "falling" and slope <= thresh: confirmed = step_def["step"]
        if direct == "stable"  and abs(slope) <= abs(thresh): confirmed = step_def["step"]

    return confirmed

# ─────────────────────────────────────────────
# CONVERGENCE SCORER
# Combines DTW confidence + historical similarity
# + slope into a single priority score.
# priority = 0.4*dtw_conf + 0.35*hist_sim + 0.25*slope_score
# ─────────────────────────────────────────────

def convergence_score(zone_id: str, dtw_conf: float) -> dict:
    """
    Compute final priority score for a zone.
    """
    sst_full  = generate_sst_history(zone_id, days=90)
    sst_last8 = np.array(sst_full[-8:])
    slope     = get_slope(zone_id)

    # Historical similarity: how much does recent pattern
    # deviate from the zone's own 90-day mean?
    zone_mean  = sst_full.mean()
    recent_dev = float(np.abs(sst_last8 - zone_mean).mean())
    hist_sim   = float(np.clip(recent_dev / 2.0, 0.0, 1.0))

    # Slope score: normalise slope to 0-1
    slope_score = float(np.clip(abs(slope) / 0.5, 0.0, 1.0))

    # Final weighted score
    priority = (0.40 * dtw_conf) + (0.35 * hist_sim) + (0.25 * slope_score)

    # Alert level thresholds
    if priority >= 0.70:
        alert_level = "HIGH"
    elif priority >= 0.45:
        alert_level = "WARN"
    else:
        alert_level = "NORMAL"

    return {
        "dtw_conf":    round(dtw_conf, 4),
        "hist_sim":    round(hist_sim, 4),
        "slope_score": round(slope_score, 4),
        "priority":    round(priority, 4),
        "alert_level": alert_level,
    }

# ─────────────────────────────────────────────
# MAIN SCORER — runs everything for one zone
# ─────────────────────────────────────────────

def score_zone(zone_id: str) -> dict:
    """Full ESG scoring pipeline for one zone."""
    match   = dtw_match(zone_id)
    step    = detect_step(zone_id, match["signature"])
    scores  = convergence_score(zone_id, match["dtw_conf"])
    sig_def = SIGNATURES[match["signature"]]
    total_steps = len(sig_def["steps"])
    step_name   = sig_def["steps"][min(step - 1, total_steps - 1)]["name"]

    return {
        "zone_id":     zone_id,
        "signature":   match["signature"],
        "dtw_dist":    match["dtw_dist"],
        "current_step": step,
        "total_steps":  total_steps,
        "step_name":    step_name,
        **scores
    }

def score_all_zones() -> list:
    """Score all 7 zones and return sorted by priority."""
    results = [score_zone(zid) for zid in ZONE_CONFIGS]
    return sorted(results, key=lambda x: x["priority"], reverse=True)


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"  AIRAVAT 3.0 — ESG Engine Output")
    print(f"{'='*70}\n")

    results = score_all_zones()

    for r in results:
        bar = "█" * int(r["priority"] * 20) + "░" * (20 - int(r["priority"] * 20))
        print(f"[{r['alert_level']:6}] {r['zone_id']}  {bar}  {r['priority']:.2f}")
        print(f"         Signature : {r['signature']}")
        print(f"         Step      : {r['current_step']}/{r['total_steps']} — {r['step_name']}")
        print(f"         DTW dist  : {r['dtw_dist']}  conf={r['dtw_conf']}  hist={r['hist_sim']}  slope={r['slope_score']}")
        print()