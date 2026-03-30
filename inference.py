"""
Inference Script for ICU Drug Titration Environment.

Runs a real LLM agent (OpenAI client) against the ICU Drug Titration
OpenEnv environment. The agent acts as an expert ICU clinical pharmacist,
making drug titration decisions step-by-step to stabilise patient vitals.

Features:
    - Modular architecture: run_task, call_llm, parse_action, fallback_action
    - Robust JSON parsing with code-fence stripping
    - MAP-based fallback policy guaranteeing stable execution
    - Detailed clinical system & user prompts
    - Clean step-by-step logging with final grading

Usage:
    # Set required environment variables
    export OPENAI_API_KEY=sk-...        # or HF_TOKEN
    export MODEL_NAME=gpt-4o-mini       # optional, defaults to gpt-4o-mini
    export API_BASE_URL=http://...      # optional, defaults to http://localhost:7860

    python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package is required. Install with: pip install openai")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Resolve API key: prefer OPENAI_API_KEY, fall back to HF_TOKEN
_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
if not _api_key:
    print("ERROR: Set OPENAI_API_KEY or HF_TOKEN environment variable.")
    sys.exit(1)

MAX_STEPS: int = 30
TASKS: List[str] = ["easy", "medium", "hard"]

# Valid action types (matches ActionType enum in models.py)
VALID_ACTION_TYPES = {"add_drug", "titrate", "remove_drug", "hold", "order_lab", "flag_physician"}

# Drug dose limits for validation & smoothing
DOSE_LIMITS: Dict[str, Tuple[float, float]] = {
    "norepinephrine": (0.01, 0.5),
    "vasopressin":    (0.01, 0.04),
    "dobutamine":     (2.0, 20.0),
    "propofol":       (5.0, 80.0),
    "fentanyl":       (25.0, 200.0),
    "insulin":        (0.5, 15.0),
}

# Tracks the last action per task for dose smoothing
_last_actions: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """\
You are a cautious, experienced ICU clinical pharmacist AI agent. You are \
managing drug titration for a critically ill patient in a simulated ICU.

## Core Philosophy: STABILITY FIRST
Your PRIMARY goal is STABILITY — not aggressive correction.
- Once a vital sign reaches its safe range, MAINTAIN it. Do NOT overcorrect.
- Prefer SMALL, INCREMENTAL dose changes. Never make large jumps.
- If the patient is currently stable (all key vitals in range), use the HOLD \
action. Unnecessary changes cause oscillations and harm.
- Avoid frequent up-down titration of the same drug. Pick a direction and \
adjust gradually.
- A patient who is "good enough" (vitals near target) is better off with no \
change than a risky adjustment.

## Safe Therapeutic Ranges
### Vitals
| Parameter | Safe Range           | Priority |
|-----------|----------------------|----------|
| MAP       | 65 – 90 mmHg        | HIGHEST  |
| HR        | 60 – 100 bpm        | HIGH     |
| SpO2      | 94 – 100 %          | HIGH     |
| RR        | 12 – 20 breaths/min | HIGH     |
| Temp      | 36.5 – 37.5 °C      | MODERATE |

### Labs
| Parameter   | Safe Range        | Priority |
|-------------|-------------------|----------|
| Glucose     | 70 – 180 mg/dL    | MODERATE |
| Creatinine  | 0.6 – 1.2 mg/dL   | LOW (cannot be directly treated) |
| Potassium   | 3.5 – 5.0 mEq/L   | MODERATE |
| Lactate     | 0.5 – 2.0 mmol/L  | MODERATE (improves with perfusion) |

## Available Drug Dose Ranges
| Drug            | Min Dose | Max Dose | Unit         | Typical Start |
|-----------------|----------|----------|--------------|---------------|
| norepinephrine  | 0.01     | 0.5      | mcg/kg/min   | 0.05 – 0.1    |
| vasopressin     | 0.01     | 0.04     | units/min    | 0.01 – 0.02   |
| dobutamine      | 2.0      | 20.0     | mcg/kg/min   | 2.5 – 5.0     |
| propofol        | 5.0      | 80.0     | mcg/kg/min   | 10 – 25       |
| fentanyl        | 25.0     | 200.0    | mcg/hr       | 25 – 50       |
| insulin         | 0.5      | 15.0     | units/hr     | 1.0 – 3.0     |

## CRITICAL Safety Rules
1. ALWAYS stay within the dose bounds listed above.
2. NEVER combine propofol + fentanyl → CRITICAL respiratory depression risk.
3. NEVER combine vasopressin + norepinephrine → CRITICAL peripheral ischemia.
4. Sedatives are DANGEROUS at high doses. Keep propofol ≤ 30 mcg/kg/min for \
safety. Keep fentanyl ≤ 75 mcg/hr unless absolutely necessary.
5. If RR drops below 12, consider REDUCING or REMOVING sedatives immediately.
6. When titrating, change dose by at most 30-50% of the current dose per step.

## Multi-Drug Strategy (Medium & Hard Tasks)
- Do NOT rely on a single drug pushed to high doses. Combine drugs at moderate \
doses for synergistic effects.
- For septic shock: use norepinephrine for MAP, insulin for glucose (start \
low: 1-3 units/hr), monitor potassium.
- For ventilated patients: prefer propofol at LOW doses (10-25 mcg/kg/min). \
Avoid combining with fentanyl.
- Monitor renal function (creatinine). Avoid aggressive insulin if potassium \
is already dropping.
- If glucose is high but potassium is low/normal, use insulin cautiously \
(≤ 3 units/hr).

## Decision Priority
1. Is patient stable? → HOLD
2. Is MAP dangerously low (< 65)? → Add/titrate vasopressor
3. Is RR dangerously low (< 10)? → Reduce/remove sedatives
4. Are other vitals out of range? → Make ONE small change
5. Are only labs out of range? → Consider gentle intervention or HOLD

## Response Format
Respond with ONLY a single JSON object. No explanation, no markdown fences:
{"action_type": "<type>", "drug": "<name>", "dose": <number>}

Valid action_type values:
- add_drug: Start a new drug infusion (requires drug + dose)
- titrate: Change dose of an active drug (requires drug + dose)
- remove_drug: Stop a drug infusion (requires drug)
- hold: Take no action this step
- order_lab: Order fresh lab results
- flag_physician: Escalate to attending physician
"""


# =============================================================================
# OPENAI CLIENT
# =============================================================================

openai_client = OpenAI(api_key=_api_key)


# =============================================================================
# HTTP CLIENT
# =============================================================================

http_client = httpx.Client(base_url=API_BASE_URL, timeout=60.0)


# =============================================================================
# HELPER: FORMAT OBSERVATION FOR LLM
# =============================================================================

def format_observation_prompt(obs: Dict[str, Any], step: int) -> str:
    """Build a rich user prompt from the current observation."""

    vitals = obs["vitals"]
    labs = obs["labs"]
    active_drugs = obs.get("active_drugs", [])
    alerts = obs.get("alerts", [])
    vitals_in_range = obs.get("vitals_in_range", {})
    labs_in_range = obs.get("labs_in_range", {})
    max_steps = obs.get("max_steps", MAX_STEPS)
    disease = obs.get("disease", "unknown")
    task_id = obs.get("task_id", "unknown")
    remaining = max_steps - step

    # Count how many vitals/labs are in range for stability summary
    vitals_ok = sum(1 for v in vitals_in_range.values() if v)
    vitals_total = len(vitals_in_range) if vitals_in_range else 5
    labs_ok = sum(1 for v in labs_in_range.values() if v)
    labs_total = len(labs_in_range) if labs_in_range else 4

    lines = [
        f"== STEP {step} / {max_steps}  |  Remaining: {remaining} steps  |  Task: {task_id}  |  Disease: {disease} ==",
        f"   Stability: {vitals_ok}/{vitals_total} vitals OK, {labs_ok}/{labs_total} labs OK",
        "",
        "--- VITALS ---",
        f"  MAP:  {vitals['map']:.1f} mmHg       {'✓ IN RANGE' if vitals_in_range.get('map') else '⚠ OUT OF RANGE (target: 65-90)'}",
        f"  HR:   {vitals['hr']:.1f} bpm         {'✓ IN RANGE' if vitals_in_range.get('hr') else '⚠ OUT OF RANGE (target: 60-100)'}",
        f"  SpO2: {vitals['spo2']:.1f} %           {'✓ IN RANGE' if vitals_in_range.get('spo2') else '⚠ OUT OF RANGE (target: 94-100)'}",
        f"  RR:   {vitals['rr']:.1f} breaths/min {'✓ IN RANGE' if vitals_in_range.get('rr') else '⚠ OUT OF RANGE (target: 12-20)'}",
        f"  Temp: {vitals['temp']:.1f} °C          {'✓ IN RANGE' if vitals_in_range.get('temp') else '⚠ OUT OF RANGE (target: 36.5-37.5)'}",
        "",
        "--- LABS ---",
        f"  Glucose:    {labs['glucose']:.1f} mg/dL   {'✓ IN RANGE' if labs_in_range.get('glucose') else '⚠ OUT OF RANGE (target: 70-180)'}",
        f"  Creatinine: {labs['creatinine']:.2f} mg/dL  {'✓ IN RANGE' if labs_in_range.get('creatinine') else '⚠ OUT OF RANGE (target: 0.6-1.2)'}",
        f"  Potassium:  {labs['potassium']:.2f} mEq/L   {'✓ IN RANGE' if labs_in_range.get('potassium') else '⚠ OUT OF RANGE (target: 3.5-5.0)'}",
        f"  Lactate:    {labs['lactate']:.2f} mmol/L    {'✓ IN RANGE' if labs_in_range.get('lactate') else '⚠ OUT OF RANGE (target: 0.5-2.0)'}",
        "",
    ]

    # Active drugs
    if active_drugs:
        lines.append("--- ACTIVE DRUGS ---")
        for d in active_drugs:
            drug_name = d['drug_name']
            dose = d['current_dose']
            unit = d['unit']
            limits = DOSE_LIMITS.get(drug_name)
            dose_info = ""
            if limits:
                pct = ((dose - limits[0]) / (limits[1] - limits[0])) * 100 if limits[1] > limits[0] else 0
                dose_info = f"  [{pct:.0f}% of max range]"
            lines.append(f"  • {drug_name}: {dose} {unit} (since step {d['step_started']}){dose_info}")
        lines.append("")
    else:
        lines.append("--- ACTIVE DRUGS: None ---")
        lines.append("")

    # Alerts / Warnings
    if alerts:
        lines.append("--- ⚠ ALERTS — READ CAREFULLY ---")
        for a in alerts:
            severity = a.get("severity", "info").upper()
            lines.append(f"  [{severity}] {a['message']}")
        lines.append("")

    # Explicit stability instruction
    lines.append("─" * 50)
    lines.append("INSTRUCTION: Decide the safest next action.")
    lines.append("• If the patient is STABLE (key vitals in range), prefer HOLD.")
    lines.append("• If vitals are out of range, make ONE small adjustment.")
    lines.append("• Do NOT change multiple drugs at once.")
    lines.append("• Keep dose changes small (≤ 30-50% of current dose).")
    if remaining <= 3:
        lines.append("• ⚠ FEW STEPS REMAINING — avoid risky changes, prefer stability.")
    lines.append("")
    lines.append("Respond with ONLY a JSON object.")

    return "\n".join(lines)


# =============================================================================
# STABILITY HEURISTIC
# =============================================================================

def stability_heuristic(obs: Dict[str, Any]) -> bool:
    """
    Check whether the patient is currently stable enough to HOLD.

    Returns True if all critical vitals are within safe ranges,
    meaning no intervention is needed this step.
    """
    vitals = obs.get("vitals", {})
    vitals_in_range = obs.get("vitals_in_range", {})

    # Primary vitals that must be in range to consider patient "stable"
    map_ok = vitals_in_range.get("map", False)
    hr_ok = vitals_in_range.get("hr", False)
    spo2_ok = vitals_in_range.get("spo2", False)
    rr_ok = vitals_in_range.get("rr", False)

    # Use tighter MAP range for stability (65–85) to leave buffer
    map_val = vitals.get("map", 0)
    map_stable = 65.0 <= map_val <= 85.0

    return map_stable and map_ok and hr_ok and spo2_ok and rr_ok


# =============================================================================
# DOSE SMOOTHING
# =============================================================================

def smooth_dose(action: Dict[str, Any], obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply dose smoothing to prevent large oscillations.

    If the agent is titrating an active drug, limit the dose change to at
    most 50% of the current dose per step. Also clamps to valid dose bounds.
    """
    action_type = action.get("action_type")
    drug = action.get("drug")
    new_dose = action.get("dose")

    if action_type != "titrate" or drug is None or new_dose is None:
        return action

    # Find current dose of this drug
    active_drugs = obs.get("active_drugs", [])
    current_dose = None
    for d in active_drugs:
        if d["drug_name"] == drug:
            current_dose = d["current_dose"]
            break

    if current_dose is not None and current_dose > 0:
        max_change = current_dose * 0.5  # Max 50% change per step
        delta = new_dose - current_dose
        if abs(delta) > max_change:
            clamped_dose = current_dose + (max_change if delta > 0 else -max_change)
            print(f"    ⊘ Dose smoothed: {drug} {new_dose:.4f} → {clamped_dose:.4f} (max 50% change)")
            new_dose = clamped_dose

    # Clamp to valid dose bounds
    if drug in DOSE_LIMITS:
        lo, hi = DOSE_LIMITS[drug]
        new_dose = max(lo, min(hi, new_dose))

    action = dict(action)  # copy
    action["dose"] = round(new_dose, 4)
    return action


# =============================================================================
# CALL LLM
# =============================================================================

def call_llm(obs: Dict[str, Any], step: int) -> Optional[Dict[str, Any]]:
    """
    Send the current observation to the LLM and return the parsed action dict.

    Before calling the LLM, checks the stability heuristic — if the patient is
    stable, returns HOLD immediately (saves time and avoids oscillations).

    After getting an LLM response, applies dose smoothing.

    Returns None if the LLM call or parsing fails entirely.
    """
    # ── Stability gate: skip LLM if patient is stable ──────────
    if stability_heuristic(obs):
        print("    ✓ Stability heuristic → HOLD (all key vitals in range)")
        return {"action_type": "hold"}

    user_prompt = format_observation_prompt(obs, step)

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content
        if raw is None:
            return None
        action = parse_action(raw.strip())
        if action is None:
            return None
        # Apply dose smoothing to prevent oscillations
        action = smooth_dose(action, obs)
        return action
    except Exception as exc:
        print(f"    ⚠ LLM call failed: {exc}")
        return None


# =============================================================================
# PARSE ACTION
# =============================================================================

def parse_action(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON action from raw LLM output.

    Handles:
        - Plain JSON
        - JSON wrapped in ```json ... ``` code fences
        - Multiple JSON objects (takes the first valid one)
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if "```" in cleaned:
        # Extract content between code fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    # Try direct parse
    try:
        action = json.loads(cleaned)
        return _validate_action(action)
    except json.JSONDecodeError:
        pass

    # Try to find first JSON object in the string
    match = re.search(r"\{[^{}]*\}", cleaned)
    if match:
        try:
            action = json.loads(match.group(0))
            return _validate_action(action)
        except json.JSONDecodeError:
            pass

    print(f"    ⚠ Failed to parse JSON from LLM output: {raw[:120]}...")
    return None


def _validate_action(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and normalise an action dict. Returns None if invalid."""
    if not isinstance(action, dict):
        return None

    action_type = action.get("action_type")
    if action_type not in VALID_ACTION_TYPES:
        print(f"    ⚠ Invalid action_type: {action_type}")
        return None

    # Build a clean action dict with only known fields
    clean: Dict[str, Any] = {"action_type": action_type}

    if action_type in ("add_drug", "titrate", "remove_drug"):
        drug = action.get("drug")
        if not drug:
            print(f"    ⚠ Missing 'drug' field for {action_type}")
            return None
        clean["drug"] = str(drug).lower()

    if action_type in ("add_drug", "titrate"):
        dose = action.get("dose")
        if dose is None:
            print(f"    ⚠ Missing 'dose' field for {action_type}")
            return None
        try:
            clean["dose"] = float(dose)
        except (TypeError, ValueError):
            print(f"    ⚠ Invalid dose value: {dose}")
            return None

    return clean


# =============================================================================
# FALLBACK POLICY
# =============================================================================

def fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic fallback policy when the LLM fails.

    Strategy (evaluated in priority order):
        1. If MAP < 65 → titrate/add norepinephrine (incremental)
        2. If RR < 10  → reduce or remove sedatives (propofol/fentanyl)
        3. Otherwise   → hold
    """
    vitals = obs.get("vitals", {})
    map_val = vitals.get("map", 75.0)
    rr_val = vitals.get("rr", 16.0)
    active_drugs = obs.get("active_drugs", [])

    # ── Priority 1: Hypotension ────────────────────────────────
    if map_val < 65.0:
        norepi_dose = None
        for d in active_drugs:
            if d["drug_name"] == "norepinephrine":
                norepi_dose = d["current_dose"]
                break

        if norepi_dose is not None:
            # Titrate up by ~30% or +0.05, whichever is smaller
            increment = min(norepi_dose * 0.3, 0.05)
            new_dose = min(0.5, round(norepi_dose + increment, 4))
            return {"action_type": "titrate", "drug": "norepinephrine", "dose": new_dose}
        else:
            return {"action_type": "add_drug", "drug": "norepinephrine", "dose": 0.1}

    # ── Priority 2: Respiratory depression ─────────────────────
    if rr_val < 10.0:
        # Try to reduce or remove a sedative
        for d in active_drugs:
            if d["drug_name"] in ("propofol", "fentanyl"):
                current = d["current_dose"]
                lo = DOSE_LIMITS.get(d["drug_name"], (0, 0))[0]
                # If already near minimum, remove entirely
                if current <= lo * 1.5:
                    return {"action_type": "remove_drug", "drug": d["drug_name"]}
                # Otherwise reduce by 30%
                new_dose = round(max(lo, current * 0.7), 4)
                return {"action_type": "titrate", "drug": d["drug_name"], "dose": new_dose}

    # ── Default: Hold ──────────────────────────────────────────
    return {"action_type": "hold"}


# =============================================================================
# RUN SINGLE TASK
# =============================================================================

def run_task(task_id: str) -> Dict[str, Any]:
    """
    Execute a single task (easy / medium / hard) against the environment.

    Returns a results dict with score, total_reward, steps, and breakdown.
    """
    session_id = f"inference-{task_id}-{uuid.uuid4().hex[:8]}"

    print(f"\n{'=' * 60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"  Session: {session_id}")
    print(f"{'=' * 60}")

    # ── Reset ──────────────────────────────────────────────────
    try:
        res = http_client.post(
            "/reset",
            json={"task_id": task_id},
            params={"session_id": session_id},
        )
        res.raise_for_status()
        data = res.json()
    except Exception as exc:
        print(f"  ✘ Reset failed: {exc}")
        return {"score": 0.0, "total_reward": 0.0, "steps": 0, "error": str(exc)}

    obs = data["observation"]
    session_id = data.get("session_id", session_id)
    total_reward = 0.0
    step = 0
    done = False

    # ── Step Loop ──────────────────────────────────────────────
    while not done and step < MAX_STEPS:
        current_step = obs.get("current_step", step)

        # Get action from LLM
        action = call_llm(obs, current_step)

        # Fall back if LLM returned nothing valid
        if action is None:
            action = fallback_action(obs)
            print(f"  Step {step + 1}: [FALLBACK] → {json.dumps(action)}")
        else:
            print(f"  Step {step + 1}: Action → {json.dumps(action)}")

        # Execute step
        try:
            res = http_client.post(
                "/step",
                json=action,
                params={"session_id": session_id},
            )
            res.raise_for_status()
            step_data = res.json()
        except Exception as exc:
            print(f"    ✘ Step API error: {exc}")
            # On API error, try fallback then retry once
            action = fallback_action(obs)
            try:
                res = http_client.post(
                    "/step",
                    json=action,
                    params={"session_id": session_id},
                )
                res.raise_for_status()
                step_data = res.json()
            except Exception as exc2:
                print(f"    ✘ Step retry also failed: {exc2}")
                break

        reward_val = step_data["reward"]["value"]
        total_reward += reward_val
        obs = step_data["observation"]
        done = step_data["done"]
        step += 1

        terminated = step_data.get("terminated", False)
        truncated = step_data.get("truncated", False)

        print(f"    Reward → {reward_val:+.4f}  |  Done → {done}"
              f"{'  [TERMINATED]' if terminated else ''}"
              f"{'  [TRUNCATED]' if truncated else ''}")

    # ── Grade ──────────────────────────────────────────────────
    score = 0.0
    breakdown: Dict[str, Any] = {}
    done_reason: Optional[str] = None

    try:
        res = http_client.get("/grader", params={"session_id": session_id})
        res.raise_for_status()
        grade = res.json()
        score = grade["score"]
        breakdown = grade.get("breakdown", {})
        done_reason = grade.get("done_reason")
    except Exception as exc:
        print(f"  ⚠ Grader error: {exc}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n  {'─' * 40}")
    print(f"  Score:        {score:.4f}")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Steps:        {step}")
    if done_reason:
        print(f"  Done Reason:  {done_reason}")
    if breakdown:
        print(f"  Breakdown:")
        for k, v in breakdown.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"  {'─' * 40}")

    return {
        "score": score,
        "total_reward": round(total_reward, 4),
        "steps": step,
        "done_reason": done_reason,
        "breakdown": breakdown,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the LLM inference agent across all three tasks."""
    start_time = time.time()

    print("╔" + "═" * 58 + "╗")
    print("║  ICU Drug Titration — LLM Inference Agent                ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Model:    {MODEL_NAME:<47}║")
    print(f"║  Server:   {API_BASE_URL:<47}║")
    print(f"║  Tasks:    {', '.join(TASKS):<47}║")
    print("╚" + "═" * 58 + "╝")

    # Verify server is reachable
    try:
        health = http_client.get("/health")
        health.raise_for_status()
        print(f"\n✓ Server healthy: {health.json()}")
    except Exception as exc:
        print(f"\n✘ Cannot reach server at {API_BASE_URL}: {exc}")
        print("  Make sure the FastAPI server is running:")
        print("    python app.py")
        sys.exit(1)

    results: Dict[str, Dict[str, Any]] = {}

    for task_id in TASKS:
        results[task_id] = run_task(task_id)

    elapsed = time.time() - start_time

    # ── Final Report ───────────────────────────────────────────
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║                    FINAL RESULTS                         ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  {'Task':<10} {'Score':>8} {'Reward':>10} {'Steps':>7}          ║")
    print("║  " + "─" * 37 + "                   ║")

    total_score = 0.0
    for task_id in TASKS:
        r = results[task_id]
        total_score += r["score"]
        print(f"║  {task_id:<10} {r['score']:>8.4f} {r['total_reward']:>+10.4f} {r['steps']:>7}          ║")

    avg_score = total_score / len(TASKS) if TASKS else 0.0
    print("║  " + "─" * 37 + "                   ║")
    print(f"║  {'AVERAGE':<10} {avg_score:>8.4f}                              ║")
    print(f"║  Time elapsed: {elapsed:.1f}s                                  ║")
    print("╚" + "═" * 58 + "╝")

    # Exit with non-zero if average score is 0
    if avg_score == 0.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
