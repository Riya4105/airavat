from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from esg_engine import score_zone, score_all_zones
from simulate import get_last_n_days, get_sst_delta, get_slope, ZONE_CONFIGS

app = FastAPI(title="AIRAVAT 3.0 — ESG API", version="1.0.0")

# ─────────────────────────────────────────────
# CORS — allows your GitHub Pages frontend
# to call this API from the browser
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "system": "AIRAVAT 3.0", "version": "1.0.0"}


@app.get("/zones")
def get_all_zones():
    """
    Returns ESG scores for all 7 zones, sorted by priority.
    This is what the frontend calls on page load to colour the map.
    """
    results = score_all_zones()
    enriched = []

    for r in results:
        zone_id = r["zone_id"]
        enriched.append({
            **r,
            "sst_history": get_last_n_days(zone_id, 8),
            "sst_delta":   get_sst_delta(zone_id),
            "slope":       get_slope(zone_id),
        })

    return {"zones": enriched, "count": len(enriched)}


@app.get("/zones/{zone_id}")
def get_zone(zone_id: str):
    """
    Returns full ESG score + SST data for one zone.
    Called when user clicks a zone marker on the map.
    """
    zone_id = zone_id.upper()
    if zone_id not in ZONE_CONFIGS:
        return {"error": f"Zone {zone_id} not found. Valid: {list(ZONE_CONFIGS.keys())}"}

    result = score_zone(zone_id)
    result["sst_history"] = get_last_n_days(zone_id, 8)
    result["sst_delta"]   = get_sst_delta(zone_id)
    result["slope"]       = get_slope(zone_id)

    return result


@app.get("/health")
def health():
    return {"status": "healthy"}