"""Score baseline generations."""

from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_user_histories(gen_results: dict) -> dict[str, str]:
    """Recover stored user histories."""
    histories = {}
    for user_id, user_data in gen_results.items():
        threads = user_data.get("test_targets") or user_data.get("test_threads", [])
        for thread_data in threads:
            user_history = str(thread_data.get("user_history", "") or "")
            if user_history:
                histories[str(user_id)] = user_history
                break
    return histories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retained baseline generations")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=("qwen3-8b", "qwen3.5-397b", "gpt-5"),
        help="Baseline model to evaluate",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="all",
        choices=["turing", "all", "sim", "specificity"],
    )
    parser.add_argument("--input", type=str, required=True, help="Path to generation pickle")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: results/sft_eval)")
    parser.add_argument("--eval_num", type=int, default=None, help="Only evaluate the first N users")
    parser.add_argument("--max_workers", type=int, default=100, help="Max concurrent judge API calls")
    return parser.parse_args()


def strip_generation_suffix(path: str) -> str:
    basename = os.path.basename(path).replace(".pkl", "")
    gen_idx = basename.rfind("_gen")
    if gen_idx != -1:
        basename = basename[:gen_idx]
    return basename


def main():
    args = parse_args()
    from eval.metrics import (
        sim_judge_generate_results,
        specificity_judge_generate_results,
        turing_test_generate_results,
    )

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_prefix = strip_generation_suffix(input_path)
    output_dir = args.output_dir or os.path.join("results", "sft_eval")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading generations: {input_path}")
    with open(input_path, "rb") as handle:
        gen_results = pickle.load(handle)

    if args.eval_num is not None:
        before = len(gen_results)
        user_ids = list(gen_results.keys())[:args.eval_num]
        gen_results = {user_id: gen_results[user_id] for user_id in user_ids}
        print(f"Evaluating first {args.eval_num} users (of {before} total)")

    total_targets = sum(len(user.get("test_targets") or user.get("test_threads", [])) for user in gen_results.values())
    total_gens = sum(
        len(thread["generations"])
        for user in gen_results.values()
        for thread in (user.get("test_targets") or user.get("test_threads", []))
    )
    print(f"Users: {len(gen_results)}, Targets: {total_targets}, Generations: {total_gens}")

    user_histories = build_user_histories(gen_results)

    if args.eval in ("turing", "all"):
        print(f"\n{'=' * 60}")
        print("Running Turing Test evaluation...")
        print(f"{'=' * 60}")
        print(f"Built histories for {len(user_histories)} users")
        turing_results = turing_test_generate_results(
            gen_results,
            user_histories=user_histories,
            max_workers=args.max_workers,
        )
        output_path = os.path.join(output_dir, f"{output_prefix}_turing.pkl")
        with open(output_path, "wb") as handle:
            pickle.dump(turing_results, handle)
        print(f"Saved turing results to: {output_path}")

    if args.eval in ("sim", "all"):
        print(f"\n{'=' * 60}")
        print("Running Sim Judge evaluation...")
        print(f"{'=' * 60}")
        sim_results = sim_judge_generate_results(
            gen_results,
            user_histories=user_histories,
            max_workers=args.max_workers,
            include_breakdown=True,
        )
        output_path = os.path.join(output_dir, f"{output_prefix}_sim.pkl")
        with open(output_path, "wb") as handle:
            pickle.dump(sim_results, handle)
        print(f"Saved sim results to: {output_path}")

    if args.eval in ("specificity", "all"):
        print(f"\n{'=' * 60}")
        print("Running Specificity Judge evaluation...")
        print(f"{'=' * 60}")
        specificity_results = specificity_judge_generate_results(
            gen_results,
            user_histories=user_histories,
            max_workers=args.max_workers,
        )
        output_path = os.path.join(output_dir, f"{output_prefix}_specificity.pkl")
        with open(output_path, "wb") as handle:
            pickle.dump(specificity_results, handle)
        print(f"Saved specificity results to: {output_path}")


if __name__ == "__main__":
    main()
