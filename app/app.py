from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# The trained model is stored in the project-level model directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "best_model_compact.joblib"

_model = None


def get_model():
    """Load the model once and cache it, instead of reloading on every request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
              "Run the training notebook to create the compact model artifact."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# Columns the RandomForestRegressor was fitted on, in order.
FEATURE_COLUMNS = ["AT", "V", "AP", "RH"]

# Calibrated input ranges, taken directly from the training data
# (cleaned_data.csv, 9,527 hourly records). The frontend sliders are bounded
# to these so the model is never asked to extrapolate past what it has seen.
FEATURE_BOUNDS = {
    "AT": (1.81, 37.11),
    "V": (25.36, 81.56),
    "AP": (992.89, 1033.30),
    "RH": (25.56, 100.16),
}
PE_BOUNDS = (420.26, 495.76)


def build_input_frame(payload):
    row = {}
    for col in FEATURE_COLUMNS:
        if col not in payload or payload[col] in (None, ""):
            raise ValueError(f"Missing field: '{col}'.")
        try:
            value = float(payload[col])
        except (TypeError, ValueError):
            raise ValueError(f"'{col}' must be a number.")
        lo, hi = FEATURE_BOUNDS[col]
        if not (lo <= value <= hi):
            raise ValueError(f"'{col}' must be between {lo} and {hi}.")
        row[col] = [value]
    return pd.DataFrame(row, columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Combined Cycle Power Plant &mdash; Output Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --graphite-950: #14171a;
    --graphite-800: #1e2226;
    --graphite-700: #292e33;
    --line: #383e45;
    --line-bright: #4a515a;
    --teal: #2dd4bf;
    --teal-dim: #15423c;
    --copper: #e0a458;
    --copper-dim: #4a3722;
    --text: #e9e7e1;
    --text-dim: #9aa2a9;
    --text-faint: #666e76;
    --danger: #d9534f;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text);
    background: var(--graphite-950);
    min-height: 100vh;
    padding: 48px 20px 80px;
  }

  h1, .display, .mw-value {
    font-family: 'Rajdhani', sans-serif;
  }

  .wrap {
    max-width: 920px;
    margin: 0 auto;
  }

  .masthead {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 10px;
  }

  .masthead svg { flex: none; margin-top: 2px; }

  .masthead h1 {
    font-size: 30px;
    font-weight: 700;
    color: #fbfaf7;
    letter-spacing: -0.01em;
    margin: 0 0 6px;
  }

  .masthead p {
    margin: 0;
    color: var(--text-dim);
    font-size: 13.5px;
    line-height: 1.6;
    max-width: 56ch;
  }

  .tag-line {
    margin: 18px 0 30px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
    font-size: 11.5px;
    color: var(--text-faint);
    display: flex;
    gap: 22px;
    flex-wrap: wrap;
  }

  .tag-line b { color: var(--teal); font-weight: 500; }

  .board {
    display: grid;
    grid-template-columns: 1fr 300px;
    gap: 22px;
  }

  .panel {
    background: var(--graphite-800);
    border: 1px solid var(--line);
  }

  .inputs-panel { padding: 30px 32px; }

  .panel-label {
    font-size: 12px;
    color: var(--teal);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
  }

  .panel-label .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--teal);
    display: inline-block;
    margin-right: 7px;
    box-shadow: 0 0 6px 1px var(--teal);
  }

  .control { margin-bottom: 26px; }
  .control:last-child { margin-bottom: 0; }

  .control-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
  }

  .control-name {
    font-size: 13px;
    color: var(--text);
  }

  .control-code {
    color: var(--text-faint);
    font-size: 11px;
  }

  .control-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--copper);
  }

  .control-value .unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 400;
    color: var(--text-faint);
    margin-left: 3px;
  }

  input[type="range"] {
    -webkit-appearance: none;
    width: 100%;
    height: 4px;
    background: var(--line-bright);
    outline: none;
    margin: 0;
  }

  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--teal);
    border: 2px solid var(--graphite-800);
    box-shadow: 0 0 0 1px var(--teal);
    cursor: pointer;
    margin-top: -6px;
  }

  input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--teal);
    border: 2px solid var(--graphite-800);
    box-shadow: 0 0 0 1px var(--teal);
    cursor: pointer;
  }

  input[type="range"]::-webkit-slider-runnable-track {
    height: 4px;
    background: var(--line-bright);
  }

  .range-ticks {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 10px;
    color: var(--text-faint);
  }

  .readout-panel {
    padding: 28px 26px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .gauge-wrap {
    width: 100%;
    max-width: 240px;
    margin: 6px auto 4px;
  }

  .mw-value {
    font-size: 44px;
    font-weight: 700;
    color: #fbfaf7;
    margin-top: 6px;
    line-height: 1;
  }

  .mw-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--text-faint);
    margin-top: 4px;
    letter-spacing: 0.04em;
  }

  .mw-sub {
    margin-top: 18px;
    width: 100%;
    font-size: 11px;
    color: var(--text-faint);
    text-align: left;
    padding-top: 16px;
    border-top: 1px dashed var(--line);
    line-height: 1.7;
  }

  .mw-sub b { color: var(--text-dim); font-weight: 500; }

  @media (max-width: 760px) {
    .board { grid-template-columns: 1fr; }
    .inputs-panel { padding: 24px 20px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <div class="masthead">
    <svg width="46" height="46" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="24" r="19" stroke="#2dd4bf" stroke-width="2"/>
      <path d="M24 9 L24 24 L34 29" stroke="#2dd4bf" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M16 31 Q24 24 32 31" stroke="#e0a458" stroke-width="2" stroke-linecap="round" fill="none"/>
    </svg>
    <div>
      <h1>Combined cycle power plant &mdash; output predictor</h1>
      <p>Adjust ambient and turbine conditions to see predicted net hourly electrical output update live.</p>
    </div>
  </div>

  <div class="tag-line">
    <span><b>Model</b> Random Forest (50 trees, depth 12)</span>
    <span><b>Target</b> PE &middot; net hourly electrical output (MW)</span>
    <span><b>Calibrated on</b> 9,527 hourly records</span>
  </div>

  <div class="board">
    <div class="panel inputs-panel">
      <div class="panel-label"><span><span class="status-dot"></span>Plant conditions</span><span id="liveTag">LIVE</span></div>

      <div class="control">
        <div class="control-head">
          <span class="control-name">Ambient temperature <span class="control-code">AT</span></span>
          <span class="control-value"><span id="val-AT">19.66</span><span class="unit">&deg;C</span></span>
        </div>
        <input type="range" id="slider-AT" min="1.81" max="37.11" step="0.01" value="19.66">
        <div class="range-ticks"><span>1.8</span><span>37.1</span></div>
      </div>

      <div class="control">
        <div class="control-head">
          <span class="control-name">Exhaust vacuum <span class="control-code">V</span></span>
          <span class="control-value"><span id="val-V">54.29</span><span class="unit">cm Hg</span></span>
        </div>
        <input type="range" id="slider-V" min="25.36" max="81.56" step="0.01" value="54.29">
        <div class="range-ticks"><span>25.4</span><span>81.6</span></div>
      </div>

      <div class="control">
        <div class="control-head">
          <span class="control-name">Ambient pressure <span class="control-code">AP</span></span>
          <span class="control-value"><span id="val-AP">1013.24</span><span class="unit">mbar</span></span>
        </div>
        <input type="range" id="slider-AP" min="992.89" max="1033.30" step="0.01" value="1013.24">
        <div class="range-ticks"><span>992.9</span><span>1033.3</span></div>
      </div>

      <div class="control">
        <div class="control-head">
          <span class="control-name">Relative humidity <span class="control-code">RH</span></span>
          <span class="control-value"><span id="val-RH">73.33</span><span class="unit">%</span></span>
        </div>
        <input type="range" id="slider-RH" min="25.56" max="100.16" step="0.01" value="73.33">
        <div class="range-ticks"><span>25.6</span><span>100.2</span></div>
      </div>
    </div>

    <div class="panel readout-panel">
      <div class="panel-label" style="width:100%"><span><span class="status-dot" id="gaugeDot"></span>Output</span><span>MW</span></div>

      <div class="gauge-wrap">
        <svg viewBox="0 0 200 120" id="gaugeSvg">
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#292e33" stroke-width="14" stroke-linecap="round"/>
          <path id="gaugeArc" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#2dd4bf" stroke-width="14" stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="125.6"/>
          <line id="gaugeNeedle" x1="100" y1="100" x2="100" y2="34" stroke="#e0a458" stroke-width="3" stroke-linecap="round" transform="rotate(0 100 100)"/>
          <circle cx="100" cy="100" r="5" fill="#e0a458"/>
        </svg>
      </div>

      <div class="mw-value" id="peValue">454.3</div>
      <div class="mw-unit">MW &middot; predicted net output</div>

      <div class="mw-sub">
        Gauge range reflects the plant's observed output band in training data: <b>420.3&ndash;495.8 MW</b>. Sliders are capped to the calibrated input range the model was trained on, so predictions stay within that envelope.
      </div>
    </div>
  </div>

</div>

<script>
const FEATURES = ['AT', 'V', 'AP', 'RH'];
const PE_MIN = 420.26, PE_MAX = 495.76;

const sliders = {};
const valueLabels = {};
FEATURES.forEach(f => {
  sliders[f] = document.getElementById('slider-' + f);
  valueLabels[f] = document.getElementById('val-' + f);
});

const peValue = document.getElementById('peValue');
const gaugeNeedle = document.getElementById('gaugeNeedle');
const gaugeArc = document.getElementById('gaugeArc');
const gaugeDot = document.getElementById('gaugeDot');
const liveTag = document.getElementById('liveTag');

let debounceTimer = null;

function currentPayload() {
  const payload = {};
  FEATURES.forEach(f => { payload[f] = parseFloat(sliders[f].value); });
  return payload;
}

function updateLabels() {
  FEATURES.forEach(f => {
    valueLabels[f].textContent = parseFloat(sliders[f].value).toFixed(2);
  });
}

function setGauge(pe) {
  const clamped = Math.max(PE_MIN, Math.min(PE_MAX, pe));
  const frac = (clamped - PE_MIN) / (PE_MAX - PE_MIN);

  // needle sweeps -90deg (min) to +90deg (max) across the semicircle
  const angle = -90 + frac * 180;
  gaugeNeedle.setAttribute('transform', 'rotate(' + angle.toFixed(2) + ' 100 100)');

  // arc fill: full arc length is 251.2 (semicircle, r=80); offset counts down from that
  const arcLen = 251.2;
  gaugeArc.setAttribute('stroke-dashoffset', (arcLen * (1 - frac)).toFixed(1));

  peValue.textContent = pe.toFixed(1);
}

async function runPrediction() {
  liveTag.style.color = '#e0a458';
  liveTag.textContent = 'UPDATING';
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(currentPayload())
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.error || 'Prediction failed');

    setGauge(result.prediction);
    gaugeDot.style.background = '#2dd4bf';
    gaugeDot.style.boxShadow = '0 0 6px 1px #2dd4bf';
    liveTag.style.color = '#2dd4bf';
    liveTag.textContent = 'LIVE';
  } catch (err) {
    gaugeDot.style.background = '#d9534f';
    gaugeDot.style.boxShadow = '0 0 6px 1px #d9534f';
    liveTag.style.color = '#d9534f';
    liveTag.textContent = 'ERROR';
    console.error(err);
  }
}

function scheduleUpdate() {
  updateLabels();
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runPrediction, 150);
}

FEATURES.forEach(f => {
  sliders[f].addEventListener('input', scheduleUpdate);
});

// initial render
updateLabels();
runPrediction();
</script>
</body>
</html>"""


@app.get("/")
def index():
    return INDEX_HTML


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        frame = build_input_frame(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    prediction = float(model.predict(frame)[0])
    return jsonify({"prediction": round(prediction, 2)})


if __name__ == "__main__":
    app.run(debug=True)
