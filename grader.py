"""
Deterministic Graders for ICU Drug Titration Environment.

Provides grading functions for easy, medium, and hard tasks.
All graders are pure functions: same trajectory → same score, always.
Output is always between 0.0 and 1.0.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from models import State, StepRecord
from pharmacology_constants import (
    LAB_SAFE_RANGES,
    VITAL_SAFE_RANGES,
)


def _dose_smoothness_score(trajectory: list) -> float:
    """
    Calculate dose smoothness score based on variance of dose changes.

    Returns 0.5 (neutral) if trajectory has fewer than 2 steps.
    Otherwise, computes 1 - normalized variance of dose changes
    to produce varying scores for different trajectories.
    """
    if len(trajectory) < 2:
        return 0.5  # neutral score, not 0.0, to avoid always-same-score DQ

    # Collect the total dose across all active drugs at each step
    doses = []
    for record in trajectory:
        active = record.observation.active_drugs
        total_dose = 0.0
        for drug_info in active:
            total_dose += drug_info.current_dose
        doses.append(total_dose)

    # Calculate dose changes between consecutive steps
    dose_changes = []
    for i in range(1, len(doses)):
        dose_changes.append(abs(doses[i] - doses[i - 1]))

    if not dose_changes:
        return 0.5

    avg_change = sum(dose_changes) / len(dose_changes)
    # Normalize: small changes (<0.05) get high score, large (>0.1) get low
    smoothness = max(0.0, 1.0 - avg_change / 0.1)
    return round(smoothness, 4)


def grade_trajectory(state: State) -> Tuple[float, Dict[str, float]]:
    """
    Grade a completed trajectory based on task difficulty.

    Args:
        state: Full episode state including history.

    Returns:
        Tuple of (score, breakdown) where score is in [0.0, 1.0].
    """
    task_id = state.task_id

    if task_id == "easy":
        return grade_easy(state)
    elif task_id == "medium":
        return grade_medium(state)
    elif task_id == "hard":
        return grade_hard(state)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def grade_easy(state: State) -> Tuple[float, Dict[str, float]]:
    """
    Grade the easy task: MAP control with norepinephrine.

    Scoring breakdown:
    - 70% weight: Percentage of steps where MAP is in safe range (65-90)
    - 15% weight: Penalty for episode ending prematurely (death/flag)
    - 15% weight: Smoothness bonus (small dose changes preferred)

    Args:
        state: Full episode state.

    Returns:
        Tuple of (score, breakdown) with score in [0.0, 1.0].
    """
    breakdown = {}
    history = state.history

    if not history:
        return 0.0, {"error": "no_steps"}

    # --- MAP in range score (70% weight) ---
    map_lo, map_hi = VITAL_SAFE_RANGES["map"]
    steps_in_range = 0
    total_steps = len(history)

    for record in history:
        map_val = record.observation.vitals.map
        if map_lo <= map_val <= map_hi:
            steps_in_range += 1

    map_score = steps_in_range / total_steps
    breakdown["map_in_range"] = round(map_score, 4)

    # --- Completion bonus (15% weight) ---
    if state.done_reason == "horizon_reached":
        completion_score = 1.0
    elif state.done_reason and "flag_physician" in state.done_reason:
        completion_score = 0.3
    elif state.done_reason and "patient_death" in state.done_reason:
        completion_score = 0.0
    else:
        completion_score = 0.5
    breakdown["completion"] = round(completion_score, 4)

    # --- Smoothness bonus (15% weight) ---
    smoothness_score = _dose_smoothness_score(history)
    breakdown["smoothness"] = round(smoothness_score, 4)

    # --- Final weighted score ---
    score = (
        0.70 * map_score
        + 0.15 * completion_score
        + 0.15 * smoothness_score
    )
    score = round(max(0.0, min(1.0, score)), 4)

    return score, breakdown


def grade_medium(state: State) -> Tuple[float, Dict[str, float]]:
    """
    Grade the medium task: Multi-drug ventilated sedation.

    Scoring breakdown:
    - 40% weight: Percentage of steps with primary vitals (MAP, HR, RR) in range
    - 20% weight: Interaction penalty (% of steps without interactions)
    - 20% weight: Completion bonus
    - 20% weight: Lab stability

    Args:
        state: Full episode state.

    Returns:
        Tuple of (score, breakdown) with score in [0.0, 1.0].
    """
    breakdown = {}
    history = state.history

    if not history:
        return 0.0, {"error": "no_steps"}

    total_steps = len(history)
    primary_vitals = ["map", "hr", "rr"]

    # --- Primary vitals in range (40% weight) ---
    vitals_scores = {v: 0 for v in primary_vitals}
    for record in history:
        vir = record.observation.vitals_in_range
        for v in primary_vitals:
            if vir.get(v, False):
                vitals_scores[v] += 1

    vital_pcts = {v: count / total_steps for v, count in vitals_scores.items()}
    avg_vital_score = sum(vital_pcts.values()) / len(primary_vitals)
    breakdown["vitals_in_range"] = round(avg_vital_score, 4)

    # --- Interaction penalty (20% weight) ---
    steps_with_interaction = 0
    for record in history:
        for alert in record.observation.alerts:
            if alert.source == "drug_interaction":
                steps_with_interaction += 1
                break

    interaction_free_pct = 1.0 - (steps_with_interaction / total_steps)
    breakdown["interaction_free"] = round(interaction_free_pct, 4)

    # --- Completion bonus (20% weight) ---
    if state.done_reason == "horizon_reached":
        completion_score = 1.0
    elif state.done_reason and "flag_physician" in state.done_reason:
        completion_score = 0.3
    elif state.done_reason and "patient_death" in state.done_reason:
        completion_score = 0.0
    else:
        completion_score = 0.5
    breakdown["completion"] = round(completion_score, 4)

    # --- Lab stability (20% weight) ---
    lab_keys = ["glucose", "creatinine", "potassium", "lactate"]
    lab_in_range_counts = {k: 0 for k in lab_keys}
    for record in history:
        lir = record.observation.labs_in_range
        for k in lab_keys:
            if lir.get(k, False):
                lab_in_range_counts[k] += 1

    lab_pcts = {k: count / total_steps for k, count in lab_in_range_counts.items()}
    avg_lab_score = sum(lab_pcts.values()) / len(lab_keys)
    breakdown["lab_stability"] = round(avg_lab_score, 4)

    # --- Final weighted score ---
    score = (
        0.40 * avg_vital_score
        + 0.20 * interaction_free_pct
        + 0.20 * completion_score
        + 0.20 * avg_lab_score
    )
    score = round(max(0.0, min(1.0, score)), 4)

    return score, breakdown


def grade_hard(state: State) -> Tuple[float, Dict[str, float]]:
    """
    Grade the hard task: Septic shock with renal failure.

    Scoring breakdown:
    - 30% weight: All 5 vitals in range across episode
    - 20% weight: All 4 labs in range (especially creatinine + potassium)
    - 15% weight: Interaction-free steps
    - 15% weight: Completion
    - 10% weight: Renal-safe drug management (creatinine not worsening)
    - 10% weight: Lactate clearance bonus

    Args:
        state: Full episode state.

    Returns:
        Tuple of (score, breakdown) with score in [0.0, 1.0].
    """
    breakdown = {}
    history = state.history

    if not history:
        return 0.0, {"error": "no_steps"}

    total_steps = len(history)
    all_vitals = ["map", "hr", "spo2", "rr", "temp"]
    all_labs = ["glucose", "creatinine", "potassium", "lactate"]

    # --- All vitals in range (30% weight) ---
    vital_counts = {v: 0 for v in all_vitals}
    for record in history:
        vir = record.observation.vitals_in_range
        for v in all_vitals:
            if vir.get(v, False):
                vital_counts[v] += 1

    vital_pcts = {v: count / total_steps for v, count in vital_counts.items()}
    avg_vital_score = sum(vital_pcts.values()) / len(all_vitals)
    breakdown["vitals_in_range"] = round(avg_vital_score, 4)

    # --- All labs in range (20% weight) ---
    lab_counts = {k: 0 for k in all_labs}
    for record in history:
        lir = record.observation.labs_in_range
        for k in all_labs:
            if lir.get(k, False):
                lab_counts[k] += 1

    lab_pcts = {k: count / total_steps for k, count in lab_counts.items()}
    avg_lab_score = sum(lab_pcts.values()) / len(all_labs)
    breakdown["labs_in_range"] = round(avg_lab_score, 4)

    # --- Interaction-free (15% weight) ---
    interaction_steps = 0
    critical_steps = 0
    for record in history:
        has_interaction = False
        for alert in record.observation.alerts:
            if alert.source == "drug_interaction":
                has_interaction = True
                if alert.severity == "critical":
                    critical_steps += 1
                break
        if has_interaction:
            interaction_steps += 1

    interaction_free = 1.0 - (interaction_steps / total_steps)
    critical_penalty = critical_steps / total_steps
    interaction_score = max(0.0, interaction_free - 0.5 * critical_penalty)
    breakdown["interaction_free"] = round(interaction_score, 4)

    # --- Completion (15% weight) ---
    if state.done_reason == "horizon_reached":
        completion_score = 1.0
    elif state.done_reason and "flag_physician" in state.done_reason:
        completion_score = 0.2
    elif state.done_reason and "patient_death" in state.done_reason:
        completion_score = 0.0
    else:
        completion_score = 0.5
    breakdown["completion"] = round(completion_score, 4)

    # --- Renal-safe management (10% weight) ---
    # Creatinine should not worsen from baseline 2.8
    if len(history) >= 2:
        first_creat = history[0].observation.labs.creatinine
        last_creat = history[-1].observation.labs.creatinine
        creat_change = last_creat - first_creat
        if creat_change <= 0:
            renal_score = 1.0  # Improved or stable
        elif creat_change < 0.5:
            renal_score = 0.7
        elif creat_change < 1.0:
            renal_score = 0.4
        else:
            renal_score = 0.1
    else:
        renal_score = 0.5
    breakdown["renal_management"] = round(renal_score, 4)

    # --- Lactate clearance (10% weight) ---
    if len(history) >= 2:
        first_lactate = history[0].observation.labs.lactate
        last_lactate = history[-1].observation.labs.lactate
        lactate_change = first_lactate - last_lactate  # Positive = clearing
        if lactate_change > 2.0:
            lactate_score = 1.0
        elif lactate_change > 1.0:
            lactate_score = 0.8
        elif lactate_change > 0:
            lactate_score = 0.5
        else:
            lactate_score = 0.2
    else:
        lactate_score = 0.5
    breakdown["lactate_clearance"] = round(lactate_score, 4)

    # --- Final weighted score ---
    score = (
        0.30 * avg_vital_score
        + 0.20 * avg_lab_score
        + 0.15 * interaction_score
        + 0.15 * completion_score
        + 0.10 * renal_score
        + 0.10 * lactate_score
    )
    score = round(max(0.0, min(1.0, score)), 4)

    return score, breakdown
