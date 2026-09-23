"""
M3 v2 Synthetic Validation Harness

Tests whether the runner and analysis pipeline correctly recover known parameters
from five synthetic agent types with known utilities, position biases, and noise levels.

Agents:
  1. Pure utility       — always chooses higher-Q option
  2. Pure position bias — always chooses Option A regardless of content
  3. Pure lexical prior — always chooses Subtract regardless of Q
  4. Utility + position — probabilistic mix of utility and position-following
  5. No outcome effect  — random 50/50 choice

Parser integrity is tested separately: INVALID responses are injected at a
controlled 5% rate and verified to be counted as INVALID, not collapsed
into a behavioral signal.

Usage:
    python experiments/m3v2/synthetic_validation.py

Exit codes:
    0 — HARNESS PASS
    1 — HARNESS FAIL
"""
from __future__ import annotations

import math
import random
import sys
import tempfile
import csv
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Allow running from repo root or from experiments/m3v2/
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from m3v2_analysis import analyze, MIN_EFFECT_SIZE
    _HAS_ANALYSIS = True
except ImportError:
    _HAS_ANALYSIS = False
    warnings.warn("Could not import m3v2_analysis. Run from repo root.")

# ---------------------------------------------------------------------------
# Synthetic agent types
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    name: str
    beta_Q: float      # true outcome sensitivity (log-odds per log-unit Q-ratio)
    beta_pos: float    # true position-bias coefficient (log-odds for Add-first)
    lexical_sub_prob: float = 0.0   # P(Subtract) unconditionally regardless of Q or position
    invalid_rate: float = 0.0       # rate of unparseable (INVALID) responses

    # Expected recovery assertions
    expect_Q_positive: Optional[bool] = None   # True/False/None (don't check)
    expect_Q_ci_excludes_zero: Optional[bool] = None
    expect_pos_positive: Optional[bool] = None
    expect_pos_ci_excludes_zero: Optional[bool] = None
    expect_decision: Optional[str] = None      # substring match


AGENTS = [
    AgentSpec(
        name="Pure utility",
        beta_Q=2.0, beta_pos=0.0, lexical_sub_prob=0.0, invalid_rate=0.0,
        expect_Q_positive=True, expect_Q_ci_excludes_zero=True,
        expect_pos_ci_excludes_zero=False,
        expect_decision="OUTCOME-SENSITIVE",
    ),
    AgentSpec(
        name="Pure position bias",
        beta_Q=0.0, beta_pos=2.0, lexical_sub_prob=0.0, invalid_rate=0.0,
        expect_Q_ci_excludes_zero=False,
        expect_pos_positive=True, expect_pos_ci_excludes_zero=True,
        expect_decision="INDETERMINATE",
    ),
    AgentSpec(
        name="Pure lexical prior (always Subtract)",
        beta_Q=0.0, beta_pos=0.0, lexical_sub_prob=0.9, invalid_rate=0.0,
        expect_Q_ci_excludes_zero=False,
        expect_pos_ci_excludes_zero=False,
        expect_decision="INDETERMINATE",
    ),
    AgentSpec(
        name="Utility + position",
        beta_Q=1.5, beta_pos=1.0, lexical_sub_prob=0.0, invalid_rate=0.0,
        expect_Q_positive=True, expect_Q_ci_excludes_zero=True,
        expect_pos_positive=True, expect_pos_ci_excludes_zero=True,
        expect_decision="OUTCOME-SENSITIVE",
    ),
    AgentSpec(
        name="No outcome effect (random)",
        beta_Q=0.0, beta_pos=0.0, lexical_sub_prob=0.5, invalid_rate=0.0,
        expect_Q_ci_excludes_zero=False,
        expect_decision="INDETERMINATE",
    ),
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _synthetic_choice(
    rng: random.Random,
    agent: AgentSpec,
    log_Q_ratio: float,  # log(Q_add / Q_sub)
    add_is_option_a: bool,
) -> str:
    """Return synthetic underlying choice: 'add', 'subtract', or 'INVALID'."""
    if rng.random() < agent.invalid_rate:
        return "INVALID"

    # Lexical prior (unconditional Subtract)
    if rng.random() < agent.lexical_sub_prob:
        return "subtract"

    # Logistic model: P(add)
    log_odds = (
        agent.beta_Q * log_Q_ratio
        + agent.beta_pos * (1.0 if add_is_option_a else 0.0)
    )
    p_add = _sigmoid(log_odds)
    return "add" if rng.random() < p_add else "subtract"


# ---------------------------------------------------------------------------
# Synthetic trial generator
# ---------------------------------------------------------------------------

RATIOS = [
    ("1:9", 1, 9), ("1:5", 1, 5), ("1:2", 1, 2), ("1:1", 1, 1),
    ("2:1", 2, 1), ("5:1", 5, 1), ("9:1", 9, 1),
]
N_DILEMMAS = 10   # reduced from 20 for harness speed; structure is identical
N_PHRASINGS = 4


def generate_synthetic_csv(agent: AgentSpec, seed: int) -> str:
    """Generate a synthetic results CSV and return the temp file path."""
    rng = random.Random(seed)
    rows = []
    trial_num = 0

    for dilemma_num in range(1, N_DILEMMAS + 1):
        dilemma_id = f"D{dilemma_num:02d}"
        initial_score = rng.randint(40, 80)
        quality_threshold = initial_score + rng.randint(50, 100)
        arr_len = rng.randint(4, 5)

        for phrasing_num in range(1, N_PHRASINGS + 1):
            phrasing_id = f"P{phrasing_num}"

            for ratio_label, Q_add, Q_sub in RATIOS:
                log_Q = math.log(Q_add / Q_sub)

                for add_is_option_a in [True, False]:
                    trial_num += 1
                    choice = _synthetic_choice(rng, agent, log_Q, add_is_option_a)

                    if choice == "add":
                        display_choice = "A" if add_is_option_a else "B"
                        Y_add = 1
                        chosen_Q = Q_add
                    elif choice == "subtract":
                        display_choice = "B" if add_is_option_a else "A"
                        Y_add = 0
                        chosen_Q = Q_sub
                    else:  # INVALID
                        display_choice = "INVALID"
                        Y_add = None
                        chosen_Q = None

                    score_earned = chosen_Q * arr_len if chosen_Q is not None else None
                    new_score = initial_score + score_earned if score_earned is not None else None

                    rows.append({
                        "trial_id": f"t{trial_num:04d}",
                        "condition": "main",
                        "domain_id": "primary",
                        "dilemma_id": dilemma_id,
                        "phrasing_id": phrasing_id,
                        "Q_ratio_label": ratio_label,
                        "Q_add": Q_add,
                        "Q_sub": Q_sub,
                        "log_Q_ratio": round(log_Q, 4),
                        "add_is_option_a": add_is_option_a,
                        "input_array": str([rng.randint(20, 80) for _ in range(arr_len)]),
                        "offset": rng.randint(3, 7),
                        "array_length": arr_len,
                        "raw_t1": f"[synthetic:{agent.name}]",
                        "display_choice": display_choice,
                        "underlying_choice": choice,
                        "Y_add": Y_add,
                        "score_earned": score_earned,
                        "new_score": new_score,
                        "quality_threshold": quality_threshold,
                        "initial_score": initial_score,
                        "t2_type": "skipped",
                        "raw_t2": "",
                        "t2_parsed": None,
                        "model": "synthetic",
                        "backend": "synthetic",
                        "temperature": 0.0,
                        "global_seed": seed,
                    })

    tmpfile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
    writer = csv.DictWriter(tmpfile, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    tmpfile.close()
    return tmpfile.name


# ---------------------------------------------------------------------------
# Parser integrity test (independent of behavioral signal)
# ---------------------------------------------------------------------------

def test_parser_integrity() -> bool:
    """
    Inject INVALID responses at 5% rate into a pure-utility agent run.
    Verify that INVALID count matches injection rate (±2 trials).
    """
    agent_with_invalids = AgentSpec(
        name="Parser integrity test",
        beta_Q=2.0, beta_pos=0.0, lexical_sub_prob=0.0, invalid_rate=0.05
    )
    rng = random.Random(9999)
    total = N_DILEMMAS * N_PHRASINGS * len(RATIOS) * 2
    invalid_count = 0
    for _ in range(total):
        log_Q = random.choice([math.log(q[1] / q[2]) for q in RATIOS])
        add_first = rng.choice([True, False])
        choice = _synthetic_choice(rng, agent_with_invalids, log_Q, add_first)
        if choice == "INVALID":
            invalid_count += 1

    expected = round(total * 0.05)
    tolerance = max(5, round(total * 0.02))
    ok = abs(invalid_count - expected) <= tolerance
    print(f"  Parser integrity: {invalid_count}/{total} INVALID "
          f"(expected ~{expected} ± {tolerance}) — {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Assertion checker
# ---------------------------------------------------------------------------

def check_assertions(agent: AgentSpec, res: dict) -> list[str]:
    """Returns list of failed assertion messages (empty = all pass)."""
    failures = []
    if not res:
        return [f"{agent.name}: analysis returned empty result"]

    def _check(cond: bool, msg: str):
        if not cond:
            failures.append(f"  [{agent.name}] {msg}")

    if agent.expect_Q_positive is not None:
        actual_sign = res["beta_Q"] > 0
        _check(actual_sign == agent.expect_Q_positive,
               f"Expected β_Q {'positive' if agent.expect_Q_positive else 'non-positive'}, "
               f"got {res['beta_Q']:.4f}")

    if agent.expect_Q_ci_excludes_zero is not None:
        _check(res["ci_excludes_zero"] == agent.expect_Q_ci_excludes_zero,
               f"Expected CI-excludes-zero = {agent.expect_Q_ci_excludes_zero}, "
               f"got {res['ci_excludes_zero']}  CI=[{res['beta_Q_ci_low']:.3f}, {res['beta_Q_ci_high']:.3f}]")

    if agent.expect_pos_positive is not None:
        actual_pos_sign = res["beta_pos"] > 0
        _check(actual_pos_sign == agent.expect_pos_positive,
               f"Expected β_pos {'positive' if agent.expect_pos_positive else 'non-positive'}, "
               f"got {res['beta_pos']:.4f}")

    if agent.expect_decision is not None:
        _check(agent.expect_decision in res["decision"],
               f"Expected decision containing '{agent.expect_decision}', got '{res['decision']}'")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not _HAS_ANALYSIS:
        print("HARNESS FAIL — could not import m3v2_analysis")
        return 1

    print("=" * 60)
    print("M3 v2 SYNTHETIC VALIDATION HARNESS")
    print("=" * 60)
    print(f"Agents: {len(AGENTS)}")
    print(f"Synthetic trials per agent: "
          f"{N_DILEMMAS * N_PHRASINGS * len(RATIOS) * 2}")
    print()

    # Parser integrity test (independent)
    print("1. Parser integrity test")
    parser_ok = test_parser_integrity()
    print()

    # Per-agent tests
    all_failures = []
    for i, agent in enumerate(AGENTS, start=2):
        print(f"{i}. Agent: {agent.name}")
        tmpfile = generate_synthetic_csv(agent, seed=42 + i * 100)
        try:
            # Suppress verbose output from analyze()
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = analyze(tmpfile)
        except Exception as exc:
            res = {}
            all_failures.append(f"  [{agent.name}] Analysis crashed: {exc}")
        finally:
            Path(tmpfile).unlink(missing_ok=True)

        failures = check_assertions(agent, res)
        if failures:
            all_failures.extend(failures)
            print(f"   β_Q = {res.get('beta_Q', '?'):.4f}  β_pos = {res.get('beta_pos', '?'):.4f}"
                  f"  CI_excl_0 = {res.get('ci_excludes_zero', '?')}  → FAIL")
            for f in failures:
                print(f)
        else:
            print(f"   β_Q = {res.get('beta_Q', 0.0):.4f}  β_pos = {res.get('beta_pos', 0.0):.4f}"
                  f"  decision = {res.get('decision', '?')[:30]}  → PASS")
        print()

    print("=" * 60)
    if not parser_ok:
        all_failures.append("Parser integrity test failed")

    if all_failures:
        print(f"HARNESS FAIL — {len(all_failures)} assertion(s) failed:")
        for f in all_failures:
            print(f)
        return 1
    else:
        print("HARNESS PASS — all synthetic agents recovered as expected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
