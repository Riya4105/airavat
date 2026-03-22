import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from esg_engine import score_zone, score_all_zones
from simulate import get_last_n_days, get_sst_delta, get_slope, ZONE_CONFIGS

app = FastAPI(title="AIRAVAT 3.0 — ESG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# EXISTING ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "system": "AIRAVAT 3.0", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/zones")
def get_all_zones():
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
    zone_id = zone_id.upper()
    if zone_id not in ZONE_CONFIGS:
        return {"error": f"Zone {zone_id} not found."}
    result = score_zone(zone_id)
    result["sst_history"] = get_last_n_days(zone_id, 8)
    result["sst_delta"]   = get_sst_delta(zone_id)
    result["slope"]       = get_slope(zone_id)
    return result

# ─────────────────────────────────────────────
# QUERY ENDPOINT — Day 9
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

ZONE_NAMES = {
    "Z1": "Arabian Sea NW", "Z2": "Gulf of Oman",
    "Z3": "Lakshadweep Sea", "Z4": "Bay of Bengal N",
    "Z5": "Sri Lanka Coast", "Z6": "Malabar Coast",
    "Z7": "Andaman Sea"
}

SIG_LABELS = {
    "thermal_stress":  "Thermal stress",
    "hypoxic_bloom":   "Hypoxic bloom",
    "turbidity_spike": "Turbidity spike",
    "upwelling":       "Upwelling",
    "oil_slick":       "Oil slick precursor",
    "normal":          "Normal"
}

@app.post("/query")
def query(req: QueryRequest):
    """
    Takes a natural language question from the operator.
    Builds a context block from live ESG scores.
    Calls Claude API with structured system prompt.
    Returns severity → chain → explanation → action.
    """

    # Build live context from ESG engine
    all_zones   = score_all_zones()
    top3        = all_zones[:3]

    context_lines = []
    for z in all_zones:
        line = (
            f"Zone {z['zone_id']} ({ZONE_NAMES.get(z['zone_id'], z['zone_id'])}): "
            f"alert={z['alert_level']} | "
            f"signature={SIG_LABELS.get(z['signature'], z['signature'])} | "
            f"step={z['current_step']}/{z['total_steps']} | "
            f"confidence={round(z['dtw_conf']*100)}% | "
            f"priority={round(z['priority']*100)}/100 | "
            f"sst_delta={get_sst_delta(z['zone_id']):+.2f}°C | "
            f"slope={get_slope(z['zone_id']):+.4f}"
        )
        context_lines.append(line)

    context_block = "\n".join(context_lines)

    system_prompt = """You are AIRAVAT 3.0, an AI marine environmental intelligence system.
You monitor ocean zones using Ecological Signature Graph (ESG) precursor chain matching.
You detect crises BEFORE they happen by matching live sensor data against known event signatures using Dynamic Time Warping.

Your responses must ALWAYS follow this exact structure — no free-form text:

SEVERITY: [HIGH / WARN / NORMAL]
TOP ZONE: [Zone name and ID]
CHAIN STATE: [Signature name — Step X of Y — Confidence Z%]
EXPLANATION: [1-2 sentences explaining what the data shows and why it matters]
ACTION: [Specific, actionable recommendation for the operator]
---
RANKED ZONES:
1. [Zone] — [alert level] — [priority score]/100
2. [Zone] — [alert level] — [priority score]/100
3. [Zone] — [alert level] — [priority score]/100

Rules:
- Never say "I think" or "possibly" — speak with the confidence of a sensor system
- Never mention raw numbers without units
- Always name the specific ecological event
- Keep EXPLANATION under 2 sentences
- ACTION must be specific — name vessels, frequencies, agencies"""

    user_message = f"""Current zone state:
{context_block}

Operator question: {req.question}"""

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=600,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]
    )

    response_text = message.choices[0].message.content

    return {
        "question": req.question,
        "response": response_text,
        "top_zone": ZONE_NAMES.get(top3[0]["zone_id"], top3[0]["zone_id"]),
        "top_priority": top3[0]["priority"],
        "top_alert": top3[0]["alert_level"],
    }
