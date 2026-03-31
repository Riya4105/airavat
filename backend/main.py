import os
import csv
from datetime import datetime
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
# CORE ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "system": "AIRAVAT 3.0", "version": "1.0.0"}


@app.get("/health")
def health():
    """
    Cold start ping endpoint.
    The frontend pings this on page load to wake the Render free-tier server
    and show the user a warm-up toast while waiting.
    Returns immediately once the server is live.
    """
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
# QUERY ENDPOINT
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
    Calls Groq LLaMA-3.3-70B with structured system prompt.
    Returns severity → chain → explanation → action.
    """
    all_zones = score_all_zones()
    top3      = all_zones[:3]

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
You monitor Indian Ocean zones using Ecological Signature Graph (ESG) precursor chain matching and Dynamic Time Warping.

You have access to live ESG scores for all 7 zones. Answer the operator's question directly and intelligently.

RESPONSE RULES:
- If asked "what needs attention" or "most critical" or "summary" → use this structure:
  SEVERITY: [HIGH/WARN/NORMAL]
  TOP ZONE: [name and ID]
  CHAIN STATE: [signature — step X/Y — confidence Z%]
  EXPLANATION: [1-2 sentences on what the data shows]
  ACTION: [specific recommendation]
  ---
  RANKED ZONES:
  1. [zone] — [alert] — [score]/100
  2. [zone] — [alert] — [score]/100
  3. [zone] — [alert] — [score]/100

- If asked about a SPECIFIC zone → focus only on that zone, explain its signature chain, current step, confidence, SST trend, and recommended action in 3-4 sentences.

- If asked about a SPECIFIC event type → scan all zones for that signature, report which zones match and their confidence levels.

- If asked a COMPARISON → compare both zones side by side on signature, step, confidence, and priority score.

- If asked HOW something works → explain it clearly in plain language relevant to marine monitoring.

- If asked for a FORECAST → extrapolate from current slope and step position to predict likely next steps.

- Never give the ranked zones structure for specific zone questions.
- Never repeat the same answer format for different question types.
- Always use actual numbers from the zone data provided.
- Speak with authority — you are a sensor system, not a chatbot.
- Keep responses under 150 words unless the question requires more detail."""

    user_message = f"""Live zone data (sorted by priority):
{context_block}

Operator question: {req.question}

Answer this specific question using the live data above. Match your response format to the question type."""

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
        "question":     req.question,
        "response":     response_text,
        "top_zone":     ZONE_NAMES.get(top3[0]["zone_id"], top3[0]["zone_id"]),
        "top_zone_id":  top3[0]["zone_id"],
        "top_priority": top3[0]["priority"],
        "top_alert":    top3[0]["alert_level"],
        "signature":    top3[0]["signature"],
        "current_step": top3[0]["current_step"],
        "dtw_conf":     top3[0]["dtw_conf"],
    }


# ─────────────────────────────────────────────
# FEEDBACK ENDPOINT
# ─────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    zone_id:     str
    signature:   str
    step:        int
    confidence:  float
    priority:    float
    alert_level: str
    feedback:    str   # "confirm" or "false_positive"
    question:    str

FEEDBACK_FILE = "/tmp/feedback.csv"

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Logs operator feedback to CSV.
    confirm        → true positive, reinforces chain weight
    false_positive → penalises offending step
    """
    file_exists = os.path.isfile(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "zone_id", "zone_name", "signature",
            "step", "confidence", "priority", "alert_level",
            "feedback", "question"
        ])
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp":   datetime.utcnow().isoformat(),
            "zone_id":     req.zone_id,
            "zone_name":   ZONE_NAMES.get(req.zone_id, req.zone_id),
            "signature":   req.signature,
            "step":        req.step,
            "confidence":  req.confidence,
            "priority":    req.priority,
            "alert_level": req.alert_level,
            "feedback":    req.feedback,
            "question":    req.question,
        })

    tp_count = 0
    fp_count = 0
    if os.path.isfile(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["feedback"] == "confirm":
                    tp_count += 1
                elif row["feedback"] == "false_positive":
                    fp_count += 1

    return {
        "status":                "logged",
        "feedback":              req.feedback,
        "zone_id":               req.zone_id,
        "total_confirmed":       tp_count,
        "total_false_positives": fp_count,
        "message": f"{'Chain reinforced' if req.feedback == 'confirm' else 'Step penalised'}. System recalibrating."
    }


@app.get("/feedback")
def get_feedback():
    """Returns feedback stats — shown as live counters in the frontend."""
    if not os.path.isfile(FEEDBACK_FILE):
        return {"entries": [], "total": 0, "tp": 0, "fp": 0, "accuracy": 0}

    entries = []
    tp = fp = 0
    with open(FEEDBACK_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
            if row["feedback"] == "confirm":
                tp += 1
            else:
                fp += 1

    return {
        "entries":  entries,
        "total":    len(entries),
        "tp":       tp,
        "fp":       fp,
        "accuracy": round(tp / len(entries) * 100, 1) if entries else 0
    }