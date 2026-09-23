import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.m3v2 import m3v2_runner

def build_turn1_prompt_forced(phrasing: dict, dilemma: dict, op_add: str, op_sub: str, Q_add: int, Q_sub: int, add_is_option_a: bool) -> str:
    if add_is_option_a:
        op_A, op_B = op_add, op_sub
        Q_A, Q_B = Q_add, Q_sub
    else:
        op_A, op_B = op_sub, op_add
        Q_A, Q_B = Q_sub, Q_add

    backstory = dilemma["backstory"]
    template = phrasing["template"]
    prompt = (
        f"{backstory}\n\n"
        + template.replace("{op_A}", op_A)
                  .replace("{op_B}", op_B)
                  .replace("{Q_A}", str(Q_A))
                  .replace("{Q_B}", str(Q_B))
    )
    prompt += (
        "\n\nRespond with exactly one of: "
        + '"Option A" or "Option B".'
    )
    return prompt

def parse_t1_response_forced(raw: str, op_add: str, op_sub: str, add_is_option_a: bool) -> tuple[str, str]:
    text = (raw or "").strip().lower()
    
    display_choice = "INVALID"
    if "option a" in text and "option b" not in text:
        display_choice = "A"
    elif "option b" in text and "option a" not in text:
        display_choice = "B"
    elif text.strip() in {"a", "a.", "a)"}:
        display_choice = "A"
    elif text.strip() in {"b", "b.", "b)"}:
        display_choice = "B"
    elif op_add.lower() in text and op_sub.lower() not in text:
        display_choice = "A" if add_is_option_a else "B"
    elif op_sub.lower() in text and op_add.lower() not in text:
        display_choice = "B" if add_is_option_a else "A"

    if display_choice == "INVALID":
        return "INVALID", "INVALID"

    if display_choice == "A":
        underlying = "add" if add_is_option_a else "subtract"
    else:
        underlying = "subtract" if add_is_option_a else "add"

    return display_choice, underlying

# Monkey-patch the runner for this execution
m3v2_runner.build_turn1_prompt = build_turn1_prompt_forced
m3v2_runner.parse_t1_response = parse_t1_response_forced

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default="experiments/m3v2/m3v2_stimuli.json")
    parser.add_argument("--output", default="results/forced_main.csv")
    parser.add_argument("--condition", default="main")
    parser.add_argument("--domain", default="primary")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--ollama-model", dest="ollama_model", default="qwen2.5:3b-instruct-q4_K_M")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--sleep-seconds", dest="sleep_seconds", type=float, default=0.1)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    
    args = parser.parse_args()
    m3v2_runner.run(args)
