# 🏥 ICU Drug Titration Environment

> 🚀 **OpenEnv Benchmark** | Healthcare AI | Decision-Making Under Risk

> **Benchmarking AI decision-making under clinical risk, uncertainty, and multi-drug complexity.**
>
> Can an AI agent keep a critically ill patient alive for 24 hours — managing vasopressors, sedatives, and insulin — without killing them through drug interactions, overdoses, or indecision?

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧠 Why This Matters

ICU drug titration is one of the hardest real-time decision problems in medicine:

- **Life-or-death stakes.** A wrong dose of norepinephrine can crash blood pressure. Too much propofol stops breathing. Insulin overdose causes fatal hypoglycemia.
- **Compounding interactions.** Drugs don't act in isolation — propofol + fentanyl synergistically suppress respiration. Vasopressin + norepinephrine cause peripheral ischemia. The agent must reason about *combinations*, not just individual drugs.
- **Non-stationary patients.** Septic patients deteriorate hour by hour. The correct dose at step 1 may be lethal by step 12.
- **No second chances.** Unlike game benchmarks, there is no "undo". A single bad decision can terminate the episode via patient death.
- **A real problem.** Medication errors contribute to an estimated 20–30% of ICU adverse events. Automated decision support in this domain has direct clinical relevance.

Most AI benchmarks test knowledge retrieval or code generation. **This environment tests whether an AI can make safe, sequential decisions under uncertainty — the kind of reasoning required for real-world clinical deployment.**

---

## 🎯 Overview

A fully deterministic, clinically grounded OpenEnv-compliant RL environment where an AI agent acts as an ICU clinical pharmacist, managing drug titration across a simulated patient episode.

| Dimension | Details |
|-----------|---------|
| **Drugs** | 6 — norepinephrine, vasopressin, dobutamine, propofol, fentanyl, insulin |
| **Vitals** | 5 — MAP, HR, SpO₂, RR, Temperature |
| **Labs** | 4 — glucose, creatinine, potassium, lactate |
| **Diseases** | 3 — vasopressor shock, ventilated sedation, septic shock + renal failure |
| **Pharmacology** | Deterministic linear drug effects with biological noise |
| **Interactions** | 2 critical + 4 warning drug-drug interaction pairs |
| **Rewards** | Dense per-step shaping with safety penalties and terminal bonuses |
| **Grading** | Fully deterministic — same trajectory always yields the same score |

> **Dosing convention:** Drug dose ranges use clinically standard units (e.g., mcg/kg/min for vasopressors, units/hr for insulin). These are weight-normalised by convention in ICU practice. The environment assumes a standard reference patient (≈70 kg), allowing consistent simulation without explicitly modelling patient weight — a common abstraction in pharmacokinetic simulators.

## 🧪 Tasks

| Task | Disease | Drugs | Horizon | Difficulty |
|------|---------|-------|---------|------------|
| **Easy** | Vasopressor shock | norepinephrine | 12h | 🟢 Single-variable MAP control |
| **Medium** | Ventilated sedation | norepinephrine, propofol, fentanyl | 20h | 🟡 Multi-drug with interaction risk |
| **Hard** | Septic shock + renal failure | All 6 drugs | 24h | 🔴 Full complexity: renal, metabolic, hemodynamic |

---

## 🤖 AI Agent Interaction

The environment follows a standard **observe → decide → act → repeat** loop:

```
┌─────────────┐        ┌──────────────┐        ┌─────────────┐
│  Environment │──obs──▶│   AI Agent   │──act──▶│ Environment │
│   (Server)   │◀─────  │   (LLM)      │  ────▶│  (Server)   │
│              │ reward │              │ action │             │
└─────────────┘  done   └──────────────┘        └─────────────┘
```

**Each step:**

1. **Observe** — The agent receives current vitals (MAP, HR, SpO₂, RR, Temp), labs (glucose, creatinine, potassium, lactate), active drug infusions, and clinical alerts.
2. **Decide** — The agent (LLM or RL policy) selects an action: add a drug, titrate a dose, remove a drug, hold, order labs, or flag a physician.
3. **Act** — The environment processes the action deterministically: applies drug effects, computes interactions, adds biological noise, and advances the patient state by one hour.
4. **Reward** — A dense reward signal (+0.1 per stable vital, penalties for interactions/overdoses/death) guides learning.
5. **Repeat** — Until the episode ends (horizon reached, patient death, or physician flagged).

> **Gradual deterioration, not binary failure.** Patient vitals deteriorate continuously each hour according to disease-specific rates (e.g., MAP −1.5 mmHg/hr in septic shock). Three threshold tiers — *safe*, *out-of-range*, and *critical (lethal)* — provide a wide buffer zone before terminal events. For example, MAP must fall from the safe floor of 65 mmHg all the way to 30 mmHg before triggering death. The agent receives multiple warning steps with visible trends and clinical alerts, allowing timely intervention.

> The environment is **fully stateless over HTTP** — any agent (LLM, RL, heuristic) can interact via the REST API.

---

## 📊 Inference Results

Performance of a prompted LLM agent (GPT-4o-mini, stability-first strategy) across all three tasks:

| Task | Score | Interpretation |
|------|-------|---------------|
| **Easy** | ~0.70 | ✅ Stable single-variable MAP control. The agent learns to titrate norepinephrine and hold once stable. |
| **Medium** | ~0.75 – 0.86 | ✅ Multi-drug reasoning works. Agent avoids lethal propofol+fentanyl interaction and balances sedation with hemodynamics. |
| **Hard** | ~0.19 | ⚠️ Exposes real limitations. Simultaneous management of sepsis, renal failure, hyperglycemia, and electrolyte imbalance overwhelms current models. |

**What this tells us:**
- Easy and medium tasks validate that LLMs *can* make safe sequential clinical decisions when the problem is well-scoped.
- The hard task is a genuine unsolved challenge — it requires multi-variable reasoning, long-horizon planning, and understanding of cascading drug effects that current models struggle with.
- The gap between medium (~0.80) and hard (~0.19) is not a bug — it reflects the real clinical complexity gap between managing 2–3 drugs vs. 6 drugs with renal constraints.

**Observed failure pattern (Hard Task):**
- Failures cluster in steps 6–12, after initial MAP stabilisation creates false confidence.
- The agent correctly prioritises vasopressor titration but underreacts to rising temperature and lactate — the hallmarks of worsening sepsis.
- Terminal failure is typically driven by metabolic or thermal collapse (e.g., lactate > 20, temp > 41°C) rather than acute overdose.
- This pattern mirrors a known clinical pitfall: focusing on hemodynamics while neglecting systemic inflammatory progression.

---

## 🧠 Key Insights

From extensive experimentation with both heuristic and LLM agents:

1. **Stability > Aggression.** Agents that make small, incremental dose changes and hold when stable consistently outperform agents that aggressively chase target ranges. Overcorrection causes oscillations that compound over time.

2. **AI oscillation is the #1 failure mode.** Without explicit constraints, LLMs tend to titrate up one step and down the next — creating dangerous vital sign swings. Our inference script uses a stability heuristic and dose smoothing to counteract this.

3. **Drug interactions are the hidden killer.** The medium task's challenge isn't individual drug dosing — it's avoiding the propofol + fentanyl respiratory depression trap. Agents that naively add both drugs kill the patient.

4. **The hard task is genuinely hard.** With 6 drugs, 5 deteriorating vitals, 4 abnormal labs, and cascading interactions, the hard task approaches real ICU complexity. A score of ~0.19 is not a failure of the environment — it's an honest signal about the frontier of AI clinical reasoning.

5. **Deterministic grading enables fair comparison.** Because the simulator and grader are fully deterministic (same seed → same trajectory → same score), this environment can serve as a reliable benchmark across different AI approaches.

6. **This is a benchmark, not just a simulator.** The goal is not to "solve" the environment, but to measure how different AI systems behave under safety-critical constraints.

---

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
# → Opens at http://localhost:7860

# Run unit tests
python -m pytest tests/test_environment.py -v

# Run baseline evaluation
python baseline.py --mode heuristic

# Run LLM inference agent
export OPENAI_API_KEY=sk-...
python inference.py
```

### Docker

```bash
docker build -t icu-drug-titration .
docker run -p 7860:7860 icu-drug-titration
```

### HuggingFace Spaces

Deploy directly — the Dockerfile is configured for port 7860.

## 🧪 How to Evaluate

Run the full system end-to-end:

```bash
# Terminal 1: Start the environment server
python app.py

# Terminal 2: Run the LLM inference agent
python inference.py
```

This will:

- Run all 3 tasks (easy, medium, hard)
- Execute an LLM agent step-by-step against the environment
- Output deterministic scores via the grader

The environment is fully API-driven and can be evaluated with **any agent** (LLM, RL, heuristic).

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment for a new episode |
| `/step` | POST | Execute one action step |
| `/state` | GET | Full episode state with history |
| `/tasks` | GET | List available tasks |
| `/grader` | GET | Grade a completed episode |
| `/baseline` | GET | Run heuristic baseline |

### Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'
```

### Step

The `/step` endpoint accepts the `Action` model **directly** as the request body (not wrapped).

```bash
curl -X POST "http://localhost:7860/step?session_id=<SESSION_ID>" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "add_drug", "drug": "norepinephrine", "dose": 0.1}'
```

**Step Response** includes Gymnasium-style termination tracking:

```json
{
  "observation": { ... },
  "reward": { "value": 0.35, "breakdown": { ... } },
  "done": true,
  "terminated": true,
  "truncated": false,
  "info": { "done_reason": "patient_death: ..." }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `terminated` | bool | `true` if patient died or physician flagged |
| `truncated` | bool | `true` if episode ended by step limit |

## 💊 Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `add_drug` | `drug`, `dose` | Start a new drug infusion |
| `titrate` | `drug`, `dose` | Adjust dose of active drug |
| `remove_drug` | `drug` | Stop a drug infusion |
| `hold` | — | No changes this hour |
| `order_lab` | — | Refresh lab values |
| `flag_physician` | — | End episode (penalty) |

> **Design note on `flag_physician`:** This action represents clinical escalation to an attending physician. It is penalised (−1.5 reward, reduced completion score in grading) to encourage autonomous stabilisation, but is intentionally less severe than patient death (−5.0). This reflects a realistic clinical tradeoff: escalation is acceptable when the situation exceeds the agent's capability, but overuse indicates weak decision-making. An agent that always flags scores poorly; an agent that flags only when necessary demonstrates appropriate clinical judgement.

## ⚠️ Drug Interactions

### Critical (R = -1.0)
- **propofol + fentanyl** → Respiratory depression
- **vasopressin + norepinephrine** → Peripheral ischemia

### Warning (R = -0.3)
- **dobutamine + norepinephrine** → Tachycardia risk
- **propofol + norepinephrine** → Hemodynamic instability
- **insulin + propofol** → Hypoglycemia risk
- **fentanyl + propofol** → Apnea risk (combined respiratory depression)

> Sources: [FDA Prescribing Information](https://www.accessdata.fda.gov/scripts/cder/daf/) &amp; UpToDate ICU Pharmacology

## 📊 Reward Design

| Component | Value | Description |
|-----------|-------|-------------|
| Vital in range | +0.1 each | Per vital in safe range per step |
| Lab in range | +0.05 each | Per lab in safe range per step |
| Critical interaction | -1.0 | Dangerous drug combination |
| Warning interaction | -0.3 | Risky drug combination |
| Unsafe dose | -0.5 | Dose outside safe bounds |
| Terminal bonus | +2.0 | All vitals+labs stable at end |
| Flag physician | -1.5 | Giving up penalty |
| Patient death | -5.0 | Lethal vital sign reached |

> **Reward scaling:** Reward values are designed to maintain stable, interpretable learning signals across all three tasks. The magnitude hierarchy (death ≫ interaction > unsafe dose > flag > positive step reward) ensures that safety violations dominate the signal while incremental stabilisation remains consistently rewarded.

## 🏗️ Architecture

```
Project/
├── pharmacology_constants.py  # Drug params, ranges, interactions
├── models.py                  # Pydantic data contracts
├── patient_simulator.py       # Physiology simulation engine
├── icu_env.py                 # OpenEnv RL environment
├── grader.py                  # Deterministic grading
├── app.py                     # FastAPI server
├── baseline.py                # Baseline evaluation script
├── inference.py               # LLM inference agent
├── openenv.yaml               # OpenEnv manifest
├── Dockerfile                 # Container deployment
├── requirements.txt           # Dependencies
├── static/
│   └── index.html             # ICU monitoring dashboard
└── tests/
    └── test_environment.py    # Unit tests
```

## 📋 Grading

All graders are **deterministic**: same trajectory → same score, always.

- **Easy**: 70% MAP stability, 15% completion, 15% dose smoothness
- **Medium**: 40% vital stability, 20% interaction avoidance, 20% completion, 20% labs
- **Hard**: 30% all vitals, 20% all labs, 15% interaction-free, 15% completion, 10% renal management, 10% lactate clearance

## 🖥️ Dashboard

The built-in dark-themed ICU monitoring dashboard provides:
- Real-time vital signs with color-coded status indicators
- Lab values panel
- Vital trend charts (Chart.js)
- Active drug infusions panel
- Action controls (add/titrate/remove/hold/lab/flag)
- Step-by-step action log with rewards and alerts
- Episode grading display

## 📄 License

MIT License
