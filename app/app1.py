import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

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

# Display metadata for the frontend. Bounds are injected from FEATURE_BOUNDS
# so the page and the validation can never drift apart.
FEATURE_META = [
    {"code": "AT", "name": "Ambient temperature", "unit": "\u00b0C", "default": 19.66},
    {"code": "V", "name": "Exhaust vacuum", "unit": "cm Hg", "default": 54.29},
    {"code": "AP", "name": "Ambient pressure", "unit": "mbar", "default": 1013.24},
    {"code": "RH", "name": "Relative humidity", "unit": "%", "default": 73.33},
]

# Number of points per input in the sensitivity sweep.
SWEEP_POINTS = 41


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


def frontend_config():
    features = []
    for meta in FEATURE_META:
        lo, hi = FEATURE_BOUNDS[meta["code"]]
        features.append({**meta, "min": lo, "max": hi})
    return {"features": features, "pe": {"min": PE_BOUNDS[0], "max": PE_BOUNDS[1]}}


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
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bp: #0c3a63;
    --bp-panel: #0a3254;
    --bp-deep: #072a47;
    --ink: #e8f3fb;
    --ink-dim: #9cc3dd;
    --line: #cfe6f5;
    --line-soft: rgba(207, 230, 245, 0.38);
    --track: rgba(207, 230, 245, 0.22);
    --grid-minor: rgba(160, 205, 235, 0.09);
    --grid-major: rgba(160, 205, 235, 0.2);
    --cyan: #6fd0f5;
    --amber: #ffc857;
    --alert: #ff8a7a;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    padding: 28px 18px 60px;
    min-height: 100vh;
    font-family: 'Barlow', 'Segoe UI', Arial, sans-serif;
    font-size: 15px;
    color: var(--ink);
    background-color: var(--bp);
    background-image:
      linear-gradient(var(--grid-major) 1px, transparent 1px),
      linear-gradient(90deg, var(--grid-major) 1px, transparent 1px),
      linear-gradient(var(--grid-minor) 1px, transparent 1px),
      linear-gradient(90deg, var(--grid-minor) 1px, transparent 1px);
    background-size: 100px 100px, 100px 100px, 20px 20px, 20px 20px;
    font-variant-numeric: tabular-nums;
  }

  h1, h2, .display { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; }

  :focus-visible { outline: 2px solid var(--amber); outline-offset: 3px; }

  /* ---- drawing sheet ---------------------------------------------------- */
  .sheet {
    max-width: 1240px;
    margin: 0 auto;
    border: 2px solid var(--line);
    padding: 7px;
  }

  .sheet-inner {
    border: 1px solid var(--line);
    padding: 30px 30px 34px;
    background: rgba(7, 42, 71, 0.55);
  }

  .masthead {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 38px;
  }

  .masthead svg { flex: none; }

  .masthead h1 {
    margin: 0 0 6px;
    font-size: clamp(30px, 4.6vw, 50px);
    font-weight: 600;
    line-height: 1.02;
    letter-spacing: 0.01em;
  }

  .masthead p {
    margin: 0;
    max-width: 62ch;
    color: var(--ink-dim);
    line-height: 1.55;
  }

  /* ---- layout ----------------------------------------------------------- */
  .board {
    display: grid;
    grid-template-columns: 360px minmax(0, 1fr);
    gap: 34px 30px;
    align-items: start;
  }

  .col { display: flex; flex-direction: column; gap: 34px; min-width: 0; }

  .panel {
    position: relative;
    padding: 30px 24px 24px;
    border: 1px solid var(--line-soft);
    background-color: var(--bp-panel);
    background-image:
      linear-gradient(var(--grid-minor) 1px, transparent 1px),
      linear-gradient(90deg, var(--grid-minor) 1px, transparent 1px);
    background-size: 20px 20px;
  }

  .panel-title {
    position: absolute;
    top: -13px;
    left: 16px;
    margin: 0;
    padding: 0 10px;
    background: var(--bp-panel);
    color: var(--cyan);
    font-size: 17px;
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  .panel-status {
    position: absolute;
    top: -12px;
    right: 16px;
    padding: 0 10px;
    background: var(--bp-panel);
    font-size: 13px;
    color: var(--ink-dim);
    display: flex;
    align-items: center;
    gap: 7px;
  }

  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 6px var(--cyan);
  }
  .dot.busy { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
  .dot.fail { background: var(--alert); box-shadow: 0 0 6px var(--alert); }

  /* ---- controls --------------------------------------------------------- */
  .presets {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 26px;
  }

  .presets button {
    font: inherit;
    font-size: 13px;
    color: var(--ink);
    background: transparent;
    border: 1px solid var(--line-soft);
    padding: 6px 11px;
    cursor: pointer;
  }
  .presets button:hover { border-color: var(--cyan); color: var(--cyan); }

  .control { margin-bottom: 24px; }
  .control:last-child { margin-bottom: 0; }

  .control-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 8px;
  }

  .control-name { font-size: 14.5px; }
  .sym { color: var(--ink-dim); font-style: italic; margin-left: 4px; }

  .entry { display: flex; align-items: baseline; gap: 6px; }

  .entry input[type="number"] {
    width: 96px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 24px;
    font-weight: 600;
    color: var(--amber);
    text-align: right;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--line-soft);
    padding: 0 2px;
    -moz-appearance: textfield;
    appearance: textfield;
  }
  .entry input[type="number"]::-webkit-outer-spin-button,
  .entry input[type="number"]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
  .entry input[type="number"]:focus { outline: none; border-bottom-color: var(--amber); }

  .unit { font-size: 12.5px; color: var(--ink-dim); min-width: 38px; }

  input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    display: block;
    width: 100%;
    height: 22px;
    margin: 0;
    background: transparent;
    cursor: pointer;
  }
  input[type="range"]::-webkit-slider-runnable-track {
    height: 3px;
    background: linear-gradient(to right, var(--cyan) var(--pos, 50%), var(--track) var(--pos, 50%));
  }
  input[type="range"]::-moz-range-track { height: 3px; background: var(--track); }
  input[type="range"]::-moz-range-progress { height: 3px; background: var(--cyan); }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    box-sizing: border-box;
    width: 12px; height: 22px;
    margin-top: -9.5px;
    background: var(--bp-deep);
    border: 2px solid var(--amber);
    border-radius: 0;
  }
  input[type="range"]::-moz-range-thumb {
    box-sizing: border-box;
    width: 12px; height: 22px;
    background: var(--bp-deep);
    border: 2px solid var(--amber);
    border-radius: 0;
  }

  .ruler {
    height: 7px;
    margin-top: 2px;
    border-right: 1px solid var(--line-soft);
    background: repeating-linear-gradient(90deg, var(--line-soft) 0 1px, transparent 1px 10%);
  }

  .scale {
    display: flex;
    justify-content: space-between;
    margin-top: 3px;
    font-size: 12px;
    color: var(--ink-dim);
  }

  /* ---- readout ---------------------------------------------------------- */
  .gauge { width: 100%; max-width: 250px; margin: 0 auto; display: block; }

  .pe-block { text-align: center; margin-top: 2px; }

  .pe-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 70px;
    font-weight: 600;
    line-height: 1;
    color: var(--ink);
  }
  .pe-value small { font-size: 26px; color: var(--amber); margin-left: 6px; font-weight: 500; }

  .pe-caption { font-size: 13.5px; color: var(--ink-dim); margin-top: 4px; }

  .indicators { margin: 22px 0 0; padding-top: 18px; border-top: 1px dashed var(--line-soft); }

  .indicators > div { margin-bottom: 16px; }
  .indicators > div:last-child { margin-bottom: 0; }

  .indicators dt { font-size: 13px; color: var(--ink-dim); margin-bottom: 3px; }
  .indicators dd {
    margin: 0;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 25px;
    font-weight: 600;
  }
  .indicators dd.up { color: var(--cyan); }
  .indicators dd.down { color: var(--alert); }

  .band { height: 8px; margin-top: 6px; border: 1px solid var(--line-soft); position: relative; }
  .band i { position: absolute; left: 0; top: 0; bottom: 0; width: 0; background: var(--cyan); transition: width 0.3s ease; }

  .error-msg { margin: 14px 0 0; font-size: 13px; color: var(--alert); min-height: 1em; }

  /* ---- schematic -------------------------------------------------------- */
  #schematic { display: block; width: 100%; height: auto; }

  #schematic text { font-family: 'Barlow', sans-serif; font-size: 13px; fill: var(--ink); }
  #schematic .dim { fill: var(--ink-dim); font-size: 12px; }
  #schematic .sym-t { fill: var(--ink-dim); font-style: italic; }
  #schematic .val { fill: var(--amber); font-weight: 600; }
  #schematic .pe-t { font-family: 'Barlow Condensed', sans-serif; font-size: 25px; font-weight: 600; fill: var(--amber); }

  .outline {
    fill: var(--bp-deep);
    stroke: var(--line);
    stroke-width: 1.6;
    stroke-linejoin: round;
    stroke-dasharray: 1;
    stroke-dashoffset: 1;
    animation: draw 1.9s ease-out forwards;
  }
  .outline.open { fill: none; }
  .detail { fill: none; stroke: var(--line-soft); stroke-width: 1; }
  .tagbox { fill: var(--bp-deep); stroke: var(--line-soft); stroke-width: 1; }
  .leader { fill: none; stroke: var(--line-soft); stroke-width: 1; stroke-dasharray: 3 3; }

  .pipe { fill: none; stroke: rgba(207, 230, 245, 0.2); stroke-width: 7; stroke-linejoin: round; }
  .flow {
    fill: none;
    stroke-width: 2;
    stroke-dasharray: 6 8;
    animation: flow var(--flow, 1.6s) linear infinite;
  }
  .flow.cyan { stroke: var(--cyan); }
  .flow.amber { stroke: var(--amber); }

  .puff { fill: none; stroke: var(--line-soft); stroke-width: 1.2; opacity: 0; animation: rise 3.4s ease-out infinite; transform-box: fill-box; transform-origin: center; }
  .puff:nth-of-type(2) { animation-delay: 1.1s; }
  .puff:nth-of-type(3) { animation-delay: 2.2s; }

  @keyframes draw { to { stroke-dashoffset: 0; } }
  @keyframes flow { to { stroke-dashoffset: -14; } }
  @keyframes rise {
    0% { opacity: 0; transform: translateY(0) scale(0.6); }
    20% { opacity: 0.8; }
    100% { opacity: 0; transform: translateY(-24px) scale(1.5); }
  }

  .legend { display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 10px; font-size: 13px; color: var(--ink-dim); }
  .legend span { display: inline-flex; align-items: center; gap: 8px; }
  .legend i { width: 26px; height: 0; border-top: 2px dashed; }
  .legend .c { border-color: var(--cyan); }
  .legend .a { border-color: var(--amber); }

  /* ---- charts ----------------------------------------------------------- */
  .charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px 26px; }

  .chart { margin: 0; }
  .chart figcaption { font-size: 14px; margin-bottom: 4px; }
  .chart svg { display: block; width: 100%; height: auto; }
  .chart .gl { stroke: var(--grid-major); stroke-width: 1; }
  .chart .axis { stroke: var(--line-soft); stroke-width: 1.2; }
  .chart .ax { fill: var(--ink-dim); font-size: 11px; font-family: 'Barlow', sans-serif; }
  .chart .curve { fill: none; stroke: var(--cyan); stroke-width: 1.8; stroke-linejoin: round; }
  .chart .mk { stroke: var(--amber); stroke-width: 1; stroke-dasharray: 3 3; }
  .chart .pt { fill: var(--amber); stroke: var(--bp-deep); stroke-width: 1.5; }
  .chart .wait { fill: var(--ink-dim); font-size: 12px; font-family: 'Barlow', sans-serif; }

  .note { margin: 18px 0 0; font-size: 13px; color: var(--ink-dim); line-height: 1.5; max-width: 74ch; }

  /* ---- title block ------------------------------------------------------ */
  .titleblock {
    margin-top: 38px;
    border: 1.5px solid var(--line);
    background: var(--bp-deep);
  }

  .tb-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); }

  .tb-cell { padding: 11px 16px 13px; border-right: 1px solid var(--line-soft); border-bottom: 1px solid var(--line-soft); }
  .tb-cell:nth-child(3n) { border-right: 0; }
  .tb-cell dt { font-size: 12px; color: var(--ink-dim); margin-bottom: 2px; }
  .tb-cell dd { margin: 0; font-size: 15px; line-height: 1.35; }
  .tb-cell.wide { grid-column: span 2; }

  .envelope { width: 100%; border-collapse: collapse; font-size: 14px; }
  .envelope caption { text-align: left; padding: 11px 16px 8px; font-size: 12px; color: var(--ink-dim); }
  .envelope th, .envelope td { padding: 7px 16px; text-align: left; border-top: 1px solid var(--line-soft); }
  .envelope th { font-weight: 500; font-size: 12px; color: var(--ink-dim); }
  .envelope td.num, .envelope th.num { text-align: right; }

  /* ---- responsive ------------------------------------------------------- */
  @media (max-width: 1020px) {
    .board { display: flex; flex-direction: column; gap: 34px; }
    .col { display: contents; }
    #panel-readout { order: 1; }
    #panel-controls { order: 2; }
    #panel-diagram { order: 3; }
    #panel-charts { order: 4; }
  }

  @media (max-width: 620px) {
    body { padding: 14px 8px 40px; }
    .sheet-inner { padding: 20px 14px 22px; }
    .masthead { flex-direction: column; align-items: flex-start; gap: 12px; }
    .panel { padding: 28px 16px 20px; }
    .charts { grid-template-columns: 1fr; }
    .tb-grid { grid-template-columns: 1fr; }
    .tb-cell, .tb-cell:nth-child(3n) { border-right: 0; }
    .tb-cell.wide { grid-column: auto; }
    .envelope th, .envelope td { padding: 7px 10px; }
  }

  @media (prefers-reduced-motion: reduce) {
    .outline { animation: none; stroke-dashoffset: 0; }
    .flow { animation: none; }
    .puff { animation: none; opacity: 0.5; }
    .band i { transition: none; }
  }
</style>
</head>
<body>
<div class="sheet"><div class="sheet-inner">

  <header class="masthead">
    <svg width="64" height="64" viewBox="-32 -32 64 64" fill="none" stroke="#6fd0f5" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
      <circle r="28"/>
      <circle r="5"/>
      <path d="M0 -5 C-4 -14 -2 -21 0 -26 M4.3 2.5 C13 3 19 7 22.5 13 M-4.3 2.5 C-9 10 -16 13 -22.5 13"/>
    </svg>
    <div>
      <h1>Combined cycle power plant output predictor</h1>
      <p>Set the ambient and exhaust conditions. A random forest model predicts the plant's net hourly electrical output, and the diagram, gauge and charts follow.</p>
    </div>
  </header>

  <main class="board">

    <div class="col">

      <section class="panel" id="panel-controls" aria-labelledby="h-controls">
        <h2 class="panel-title" id="h-controls">Plant conditions</h2>
        <div class="presets" role="group" aria-label="Preset conditions">
          <button type="button" data-preset="reference">Reference</button>
          <button type="button" data-preset="cool">Cool day</button>
          <button type="button" data-preset="hot">Hot day</button>
        </div>
        <div id="controls"></div>
      </section>

      <section class="panel" id="panel-readout" aria-labelledby="h-readout">
        <h2 class="panel-title" id="h-readout">Predicted output</h2>
        <div class="panel-status"><span class="dot" id="statusDot"></span><span id="statusText">Live</span></div>

        <svg class="gauge" viewBox="0 0 200 122" aria-hidden="true">
          <g id="gaugeTicks" stroke="#cfe6f5" stroke-opacity="0.6"></g>
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(207,230,245,0.2)" stroke-width="12"/>
          <path id="gaugeArc" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#6fd0f5" stroke-width="12" stroke-dasharray="251.2" stroke-dashoffset="251.2"/>
          <line id="gaugeNeedle" x1="100" y1="100" x2="100" y2="36" stroke="#ffc857" stroke-width="2.6" stroke-linecap="round" transform="rotate(-90 100 100)"/>
          <circle cx="100" cy="100" r="6" fill="#072a47" stroke="#ffc857" stroke-width="2"/>
          <text x="20" y="118" text-anchor="middle" font-size="10" fill="#9cc3dd" id="gaugeMin"></text>
          <text x="180" y="118" text-anchor="middle" font-size="10" fill="#9cc3dd" id="gaugeMax"></text>
        </svg>

        <div class="pe-block" aria-live="polite">
          <div class="pe-value"><span id="peValue">&ndash;</span><small>MW</small></div>
          <div class="pe-caption">Net hourly electrical output</div>
        </div>

        <dl class="indicators">
          <div>
            <dt>Position in observed output band</dt>
            <dd id="indBand">&ndash;</dd>
            <div class="band"><i id="indBandBar"></i></div>
          </div>
          <div>
            <dt>Change from reference conditions</dt>
            <dd id="indDelta">&ndash;</dd>
          </div>
          <div>
            <dt>Headroom to observed peak</dt>
            <dd id="indHead">&ndash;</dd>
          </div>
        </dl>

        <p class="error-msg" id="errorMsg" role="alert"></p>
      </section>

    </div>

    <div class="col">

      <section class="panel" id="panel-diagram" aria-labelledby="h-diagram">
        <h2 class="panel-title" id="h-diagram">Plant layout</h2>

        <svg id="schematic" viewBox="0 -30 860 440" role="img" aria-label="Simplified layout of a single-shaft combined cycle plant showing the compressor, combustor, gas turbine, generator, steam turbine, heat recovery steam generator, condenser and stack, with live input and output values.">
          <defs>
            <marker id="arrC" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 1 L9 5 L0 9z" fill="#6fd0f5"/></marker>
            <marker id="arrA" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 1 L9 5 L0 9z" fill="#ffc857"/></marker>
          </defs>

          <!-- pipes and flows -->
          <g>
            <path class="pipe" d="M10 210 H66"/>
            <path class="flow cyan" d="M10 210 H66" marker-end="url(#arrC)"/>
            <path class="pipe" d="M200 130 V176"/>
            <path class="flow amber" d="M200 130 V176" marker-end="url(#arrA)"/>
            <path class="pipe" d="M330 160 V120 H640"/>
            <path class="flow amber" d="M330 160 V120 H640"/>
            <path class="pipe" d="M640 150 H500 V176"/>
            <path class="flow cyan" d="M640 150 H500 V176"/>
            <path class="pipe" d="M560 258 V308"/>
            <path class="flow cyan" d="M560 258 V308" marker-end="url(#arrC)"/>
            <path class="pipe" d="M650 340 H740 V182"/>
            <path class="flow cyan" d="M650 340 H740 V182"/>
            <path class="pipe" d="M750 56 V14"/>
            <path class="flow amber" d="M750 56 V14"/>
            <path class="pipe" d="M400 242 V298"/>
            <path class="flow amber" d="M400 242 V298" marker-end="url(#arrA)"/>
          </g>

          <!-- shaft -->
          <line class="outline open" pathLength="1" x1="70" y1="210" x2="570" y2="210" stroke-width="3"/>

          <!-- machines -->
          <polygon class="outline" pathLength="1" points="70,160 170,185 170,235 70,260"/>
          <rect class="outline" pathLength="1" x="170" y="180" width="60" height="60" rx="8"/>
          <polygon class="outline" pathLength="1" points="230,185 330,160 330,260 230,235"/>
          <circle class="outline" pathLength="1" cx="400" cy="210" r="32"/>
          <polygon class="outline" pathLength="1" points="470,185 570,160 570,260 470,235"/>
          <rect class="outline" pathLength="1" x="640" y="60" width="150" height="120"/>
          <rect class="outline" pathLength="1" x="735" y="10" width="30" height="50"/>
          <rect class="outline" pathLength="1" x="500" y="310" width="150" height="60"/>
          <circle class="outline" pathLength="1" cx="695" cy="340" r="12"/>

          <!-- internal detail -->
          <path class="detail" d="M652 120 H778 V135 H652 V150 H778 V165 H652"/>
          <path class="detail" d="M512 325 H638 M512 340 H638 M512 355 H638"/>
          <path class="detail" d="M689 346 L695 334 L701 346 Z"/>
          <path d="M200 230 C190 218 196 208 200 196 C204 206 212 212 210 222 C209 228 205 231 200 231 Z" fill="#ffc857" fill-opacity="0.85"/>

          <!-- rotors -->
          <g transform="translate(120 210)" fill="none" stroke="#6fd0f5" stroke-width="1.4"><g><animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="3.2s" repeatCount="indefinite"/><circle r="16"/><path d="M-16 0H16M0 -16V16M-11 -11L11 11M-11 11L11 -11"/></g></g>
          <g transform="translate(280 210)" fill="none" stroke="#6fd0f5" stroke-width="1.4"><g><animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="3.2s" repeatCount="indefinite"/><circle r="16"/><path d="M-16 0H16M0 -16V16M-11 -11L11 11M-11 11L11 -11"/></g></g>
          <g transform="translate(520 210)" fill="none" stroke="#6fd0f5" stroke-width="1.4"><g><animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="3.2s" repeatCount="indefinite"/><circle r="16"/><path d="M-16 0H16M0 -16V16M-11 -11L11 11M-11 11L11 -11"/></g></g>
          <text x="400" y="219" text-anchor="middle" style="font-family:'Barlow Condensed',sans-serif;font-size:28px;font-weight:600">G</text>

          <!-- stack gas -->
          <circle class="puff" cx="750" cy="4" r="6"/>
          <circle class="puff" cx="750" cy="4" r="6"/>
          <circle class="puff" cx="750" cy="4" r="6"/>

          <!-- component labels -->
          <text x="120" y="284" text-anchor="middle">Compressor</text>
          <text x="200" y="262" text-anchor="middle" class="dim">Combustor</text>
          <text x="280" y="284" text-anchor="middle">Gas turbine</text>
          <text x="400" y="160" text-anchor="middle">Generator</text>
          <text x="500" y="284" text-anchor="middle">Steam turbine</text>
          <text x="715" y="82" text-anchor="middle">Heat recovery</text>
          <text x="715" y="98" text-anchor="middle">steam generator</text>
          <text x="575" y="388" text-anchor="middle">Condenser</text>
          <text x="695" y="372" text-anchor="middle" class="dim">Feed pump</text>
          <text x="774" y="38" class="dim">Stack</text>
          <text x="200" y="118" text-anchor="middle" class="dim">Fuel</text>
          <text x="480" y="112" text-anchor="middle" class="dim">Hot exhaust gas</text>
          <text x="570" y="142" text-anchor="middle" class="dim">Steam</text>
          <text x="10" y="198" class="dim">Air</text>

          <!-- live input tags -->
          <path class="leader" d="M44 111 V210"/>
          <rect class="tagbox" x="10" y="22" width="164" height="26"/>
          <text x="20" y="40"><tspan class="sym-t">AT</tspan>  <tspan class="val" id="tag-AT">&ndash;</tspan> &deg;C</text>
          <rect class="tagbox" x="10" y="52" width="164" height="26"/>
          <text x="20" y="70"><tspan class="sym-t">RH</tspan>  <tspan class="val" id="tag-RH">&ndash;</tspan> %</text>
          <rect class="tagbox" x="10" y="82" width="164" height="26"/>
          <text x="20" y="100"><tspan class="sym-t">AP</tspan>  <tspan class="val" id="tag-AP">&ndash;</tspan> mbar</text>

          <path class="leader" d="M560 284 H574"/>
          <rect class="tagbox" x="574" y="270" width="122" height="28"/>
          <text x="584" y="289"><tspan class="sym-t">V</tspan>  <tspan class="val" id="tag-V">&ndash;</tspan> cm Hg</text>

          <!-- live output tag -->
          <rect class="tagbox" x="335" y="300" width="130" height="56"/>
          <text x="345" y="319" class="dim">Net output</text>
          <text x="345" y="346" class="pe-t"><tspan id="tag-PE">&ndash;</tspan><tspan style="font-size:15px" dx="5">MW</tspan></text>
        </svg>

        <div class="legend">
          <span><i class="c"></i>Air, steam and water</span>
          <span><i class="a"></i>Fuel, hot gas and electricity</span>
        </div>
        <p class="note">Fig. 1. Simplified single-shaft layout, not to scale. The flow speed follows the predicted output.</p>
      </section>

      <section class="panel" id="panel-charts" aria-labelledby="h-charts">
        <h2 class="panel-title" id="h-charts">Output against each input</h2>
        <div class="charts" id="charts"></div>
        <p class="note">Each curve sweeps one input across its calibrated range while the other three stay at the values set on the left. All four charts share one vertical scale, so slopes are directly comparable. The model is a tree ensemble, so the curves are stepped.</p>
      </section>

    </div>
  </main>

  <footer class="titleblock">
    <dl class="tb-grid">
      <div class="tb-cell wide"><dt>Project</dt><dd>Combined cycle power plant net output predictor</dd></div>
      <div class="tb-cell"><dt>Prepared by</dt><dd>Abbas</dd></div>
      <div class="tb-cell"><dt>Model</dt><dd>Random forest regressor, 50 trees, depth 12</dd></div>
      <div class="tb-cell"><dt>Dataset</dt><dd>9,527 hourly records (cleaned_data.csv)</dd></div>
      <div class="tb-cell"><dt>Target</dt><dd>PE, net hourly electrical output (MW)</dd></div>
    </dl>
    <table class="envelope">
      <caption>Calibrated input envelope. Sliders are capped to these ranges so the model never extrapolates beyond its training data.</caption>
      <thead><tr><th>Variable</th><th>Symbol</th><th>Unit</th><th class="num">Minimum</th><th class="num">Maximum</th></tr></thead>
      <tbody id="envelopeRows"></tbody>
    </table>
  </footer>

</div></div>

<script>
const CONFIG = __CONFIG__;
const FEATURES = CONFIG.features;
const PE_MIN = CONFIG.pe.min, PE_MAX = CONFIG.pe.max;
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const PRESETS = {
  reference: Object.fromEntries(FEATURES.map(f => [f.code, f.default])),
  cool: { AT: 8.0, V: 40.0, AP: 1022.0, RH: 80.0 },
  hot:  { AT: 33.0, V: 72.0, AP: 1004.0, RH: 45.0 }
};

const state = { ...PRESETS.reference };
const $ = id => document.getElementById(id);
const fmt = (v, d = 2) => Number(v).toFixed(d);

/* ---- controls --------------------------------------------------------- */
const sliders = {}, numbers = {};

FEATURES.forEach(f => {
  const box = document.createElement('div');
  box.className = 'control';
  box.innerHTML =
    '<div class="control-head">' +
      '<label class="control-name" for="num-' + f.code + '">' + f.name + '<span class="sym">' + f.code + '</span></label>' +
      '<span class="entry"><input type="number" id="num-' + f.code + '" min="' + f.min + '" max="' + f.max + '" step="0.01" inputmode="decimal">' +
      '<span class="unit">' + f.unit + '</span></span>' +
    '</div>' +
    '<input type="range" id="slider-' + f.code + '" aria-label="' + f.name + ' slider" min="' + f.min + '" max="' + f.max + '" step="0.01">' +
    '<div class="ruler"></div>' +
    '<div class="scale"><span>' + f.min + '</span><span>' + f.max + '</span></div>';
  $('controls').appendChild(box);
  sliders[f.code] = $('slider-' + f.code);
  numbers[f.code] = $('num-' + f.code);

  sliders[f.code].addEventListener('input', () => setValue(f, sliders[f.code].value));
  numbers[f.code].addEventListener('change', () => setValue(f, numbers[f.code].value));
});

function setValue(f, raw, quiet) {
  let v = parseFloat(raw);
  if (!isFinite(v)) v = state[f.code];
  v = Math.min(f.max, Math.max(f.min, v));
  v = Math.round(v * 100) / 100;
  state[f.code] = v;
  sliders[f.code].value = v;
  numbers[f.code].value = fmt(v);
  sliders[f.code].style.setProperty('--pos', ((v - f.min) / (f.max - f.min) * 100).toFixed(1) + '%');
  $('tag-' + f.code).textContent = fmt(v);
  if (!quiet) scheduleRefresh();
}

function applyPreset(values) {
  FEATURES.forEach(f => setValue(f, values[f.code], true));
  scheduleRefresh();
}

document.querySelectorAll('[data-preset]').forEach(btn => {
  btn.addEventListener('click', () => applyPreset(PRESETS[btn.dataset.preset]));
});

/* ---- envelope table --------------------------------------------------- */
$('envelopeRows').innerHTML = FEATURES.map(f =>
  '<tr><td>' + f.name + '</td><td>' + f.code + '</td><td>' + f.unit + '</td><td class="num">' + f.min + '</td><td class="num">' + f.max + '</td></tr>'
).join('') +
  '<tr><td>Net hourly electrical output</td><td>PE</td><td>MW</td><td class="num">' + PE_MIN + '</td><td class="num">' + PE_MAX + '</td></tr>';

/* ---- gauge ------------------------------------------------------------ */
(function buildGaugeTicks() {
  let s = '';
  for (let i = 0; i <= 10; i++) {
    const a = (-90 + i * 18) * Math.PI / 180;
    const major = i % 5 === 0;
    const r1 = major ? 89 : 91, r2 = 97;
    s += '<line x1="' + (100 + r1 * Math.sin(a)).toFixed(2) + '" y1="' + (100 - r1 * Math.cos(a)).toFixed(2) +
         '" x2="' + (100 + r2 * Math.sin(a)).toFixed(2) + '" y2="' + (100 - r2 * Math.cos(a)).toFixed(2) +
         '" stroke-width="' + (major ? 1.6 : 1) + '"/>';
  }
  $('gaugeTicks').innerHTML = s;
  $('gaugeMin').textContent = fmt(PE_MIN, 1);
  $('gaugeMax').textContent = fmt(PE_MAX, 1);
})();

let shownPE = null, tweenId = 0;

function applyPE(v) {
  const clamped = Math.max(PE_MIN, Math.min(PE_MAX, v));
  const frac = (clamped - PE_MIN) / (PE_MAX - PE_MIN);
  $('gaugeNeedle').setAttribute('transform', 'rotate(' + (-90 + frac * 180).toFixed(2) + ' 100 100)');
  $('gaugeArc').setAttribute('stroke-dashoffset', (251.2 * (1 - frac)).toFixed(1));
  $('peValue').textContent = fmt(v, 1);
  $('tag-PE').textContent = fmt(v, 1);
}

function showPE(target) {
  cancelAnimationFrame(tweenId);
  if (shownPE === null || reduceMotion) { shownPE = target; applyPE(target); return; }
  const start = shownPE, t0 = performance.now(), dur = 380;
  const step = now => {
    const t = Math.min(1, (now - t0) / dur);
    shownPE = start + (target - start) * (1 - Math.pow(1 - t, 3));
    applyPE(shownPE);
    if (t < 1) tweenId = requestAnimationFrame(step);
  };
  tweenId = requestAnimationFrame(step);
}

/* ---- indicators and diagram ------------------------------------------- */
let baselinePE = null;

function updateIndicators(pe) {
  const frac = Math.max(0, Math.min(1, (pe - PE_MIN) / (PE_MAX - PE_MIN)));
  $('indBand').textContent = Math.round(frac * 100) + '%';
  $('indBandBar').style.width = (frac * 100).toFixed(1) + '%';

  const delta = pe - baselinePE;
  const dd = $('indDelta');
  const sign = delta > 0.05 ? '+' : (delta < -0.05 ? '\u2212' : '');
  dd.textContent = sign + fmt(Math.abs(delta), 1) + ' MW';
  dd.className = delta > 0.05 ? 'up' : (delta < -0.05 ? 'down' : '');

  $('indHead').textContent = fmt(Math.max(0, PE_MAX - pe), 1) + ' MW';

  // Flow animation: faster dashes at higher predicted output.
  $('schematic').style.setProperty('--flow', (2.4 - 1.5 * frac).toFixed(2) + 's');
}

/* ---- sensitivity charts ----------------------------------------------- */
FEATURES.forEach(f => {
  const fig = document.createElement('figure');
  fig.className = 'chart';
  fig.innerHTML = '<figcaption>Output against ' + f.name.toLowerCase() + '</figcaption>' +
    '<svg viewBox="0 0 320 190" id="chart-' + f.code + '" role="img" aria-label="Predicted output against ' + f.name.toLowerCase() + '">' +
    '<text class="wait" x="160" y="95" text-anchor="middle">Waiting for the model</text></svg>';
  $('charts').appendChild(fig);
});

function renderCharts(data, pe) {
  let lo = Infinity, hi = -Infinity;
  FEATURES.forEach(f => data[f.code].y.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  lo = Math.min(lo, pe); hi = Math.max(hi, pe);
  const range = hi - lo;
  const step = range <= 20 ? 5 : (range <= 50 ? 10 : 20);
  const ymin = Math.floor(lo / step) * step, ymax = Math.ceil(hi / step) * step;

  const W = 320, H = 190, L = 46, R = 14, T = 14, B = 38;
  const pw = W - L - R, ph = H - T - B;

  FEATURES.forEach(f => {
    const d = data[f.code];
    const X = v => L + (v - f.min) / (f.max - f.min) * pw;
    const Y = v => T + (1 - (v - ymin) / (ymax - ymin)) * ph;
    let s = '<text class="ax" x="4" y="' + (T + 3) + '">MW</text>';

    for (let v = ymin; v <= ymax + 1e-9; v += step) {
      const y = Y(v).toFixed(1);
      s += '<line class="gl" x1="' + L + '" x2="' + (W - R) + '" y1="' + y + '" y2="' + y + '"/>' +
           '<text class="ax" x="' + (L - 6) + '" y="' + (parseFloat(y) + 4) + '" text-anchor="end">' + v + '</text>';
    }
    s += '<line class="axis" x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + (T + ph) + '"/>' +
         '<line class="axis" x1="' + L + '" x2="' + (W - R) + '" y1="' + (T + ph) + '" y2="' + (T + ph) + '"/>';

    const path = d.x.map((x, i) => (i ? 'L' : 'M') + X(x).toFixed(1) + ' ' + Y(d.y[i]).toFixed(1)).join('');
    s += '<path class="curve" d="' + path + '"/>';

    const cx = X(state[f.code]).toFixed(1), cy = Y(pe).toFixed(1);
    s += '<line class="mk" x1="' + cx + '" x2="' + cx + '" y1="' + T + '" y2="' + (T + ph) + '"/>' +
         '<circle class="pt" cx="' + cx + '" cy="' + cy + '" r="4.5"/>';

    s += '<text class="ax" x="' + L + '" y="' + (T + ph + 15) + '">' + f.min + '</text>' +
         '<text class="ax" x="' + (W - R) + '" y="' + (T + ph + 15) + '" text-anchor="end">' + f.max + '</text>' +
         '<text class="ax" x="' + (L + pw / 2) + '" y="' + (H - 6) + '" text-anchor="middle">' + f.code + ', ' + f.unit + '</text>';

    $('chart-' + f.code).innerHTML = s;
  });
}

/* ---- server round trip ------------------------------------------------ */
async function post(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  let body = {};
  try { body = await res.json(); } catch (e) { /* non-JSON error page */ }
  if (!res.ok) throw new Error(body.error || ('Request failed with status ' + res.status));
  return body;
}

function setStatus(kind, text) {
  $('statusDot').className = 'dot' + (kind === 'busy' ? ' busy' : kind === 'fail' ? ' fail' : '');
  $('statusText').textContent = text;
}

let requestId = 0, debounceTimer = null;

async function refresh() {
  const id = ++requestId;
  const payload = { ...state };
  setStatus('busy', 'Updating');
  try {
    const [pred, sweep] = await Promise.allSettled([
      post('/predict', payload),
      post('/sensitivity', payload)
    ]);
    if (id !== requestId) return;
    if (pred.status === 'rejected') throw pred.reason;

    const pe = pred.value.prediction;
    if (baselinePE === null) baselinePE = pe;
    showPE(pe);
    updateIndicators(pe);
    if (sweep.status === 'fulfilled') renderCharts(sweep.value, pe);

    $('errorMsg').textContent = '';
    setStatus('ok', 'Live');
  } catch (err) {
    if (id !== requestId) return;
    $('errorMsg').textContent = err.message;
    setStatus('fail', 'Error');
    console.error(err);
  }
}

function scheduleRefresh() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(refresh, 150);
}

/* ---- start ------------------------------------------------------------ */
if (reduceMotion && $('schematic').pauseAnimations) $('schematic').pauseAnimations();
FEATURES.forEach(f => setValue(f, state[f.code], true));
refresh();
</script>
</body>
</html>"""

INDEX_PAGE = INDEX_HTML.replace("__CONFIG__", json.dumps(frontend_config()))


@app.get("/")
def index():
    return INDEX_PAGE


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


@app.post("/sensitivity")
def sensitivity():
    """Sweep each input across its calibrated range, holding the others fixed."""
    payload = request.get_json(silent=True) or {}

    try:
        base = build_input_frame(payload).iloc[0].to_dict()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    rows = []
    axes = {}
    for col in FEATURE_COLUMNS:
        lo, hi = FEATURE_BOUNDS[col]
        step = (hi - lo) / (SWEEP_POINTS - 1)
        axes[col] = [lo + i * step for i in range(SWEEP_POINTS)]
        for x in axes[col]:
            row = dict(base)
            row[col] = x
            rows.append(row)

    predictions = model.predict(pd.DataFrame(rows, columns=FEATURE_COLUMNS))

    result = {}
    for k, col in enumerate(FEATURE_COLUMNS):
        chunk = predictions[k * SWEEP_POINTS:(k + 1) * SWEEP_POINTS]
        result[col] = {
            "x": [round(x, 2) for x in axes[col]],
            "y": [round(float(p), 2) for p in chunk],
        }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)