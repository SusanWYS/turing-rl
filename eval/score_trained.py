"""Score GRPO generations."""

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
    parser = argparse.ArgumentParser(description="Evaluate retained GRPO generations")
    parser.add_argument(
        "--eval",
        type=str,
        default="all",
        choices=["turing", "all", "sim", "specificity"],
        help="Evaluation family to run. 'all' means turing + specificity + sim.",
    )
    parser.add_argument("--input", type=str, required=True, help="Path to generation pickle from generate.py")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: results/grpo_eval)")
    parser.add_argument("--eval_num", type=int, default=None, help="Only evaluate first N users")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=int(os.getenv("PERSONA_EVAL_MAX_WORKERS", "100")),
        help="Max concurrent OpenAI API calls for judge-backed evals",
    )
    return parser.parse_args()


def resolve_eval_modes(eval_name: str) -> tuple[str, ...]:
    if eval_name == "all":
        return ("turing", "specificity", "sim")
    return (eval_name,)


def main() -> None:
    args = parse_args()
    from eval.metrics import (
        sim_judge_generate_results,
        turing_test_generate_results,
    )

    basename = os.path.basename(args.input).replace(".pkl", "")
    gen_idx = basename.rfind("_gen")
    prefix = basename[:gen_idx] if gen_idx != -1 else basename

    output_dir = args.output_dir or os.path.join("results", "grpo_eval")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading generations: {args.input}")
    with open(args.input, "rb") as handle:
        gen_results = pickle.load(handle)

    if args.eval_num:
        user_ids = list(gen_results.keys())[:args.eval_num]
        gen_results = {user_id: gen_results[user_id] for user_id in user_ids}

    total_targets = sum(len(user.get("test_targets") or user.get("test_threads", [])) for user in gen_results.values())
    total_gens = sum(
        len(thread["generations"])
        for user in gen_results.values()
        for thread in (user.get("test_targets") or user.get("test_threads", []))
    )
    print(f"Users: {len(gen_results)}, Targets: {total_targets}, Generations: {total_gens}")

    eval_modes = resolve_eval_modes(args.eval)
    user_histories = None
    if any(eval_mode in ("turing", "sim", "specificity") for eval_mode in eval_modes):
        user_histories = build_user_histories(gen_results)

    if "turing" in eval_modes:
        print(f"\n{'=' * 60}")
        print("Running Turing Test evaluation...")
        print(f"{'=' * 60}")
        print(f"Built histories for {len(user_histories)} users")
        turing_results = turing_test_generate_results(
            gen_results,
            user_histories=user_histories,
            max_workers=args.max_workers,
        )
        output_path = os.path.join(output_dir, f"{prefix}_turing.pkl")
        with open(output_path, "wb") as handle:
            pickle.dump(turing_results, handle)
        print(f"Saved: {output_path}")

    if "sim" in eval_modes:
        print(f"\n{'=' * 60}")
        print("Running Sim Judge evaluation...")
        print(f"{'=' * 60}")
        sim_results = sim_judge_generate_results(
            gen_results,
            user_histories=user_histories,
            max_workers=args.max_workers,
        )
        output_path = os.path.join(output_dir, f"{prefix}_sim.pkl")
        with open(output_path, "wb") as handle:
            pickle.dump(sim_results, handle)
        print(f"Saved: {output_path}")

    if "specificity" in eval_modes:
        from eval.metrics import specificity_judge_generate_results

        print(f"\n{'=' * 60}")
        print("Running Specificity Judge evaluation...")
        print(f"{'=' * 60}")
        specificity_results = specificity_judge_generate_results(
            gen_results,
            user_histories=user_histories,
            max_workers=args.max_workers,
        )
        output_path = os.path.join(output_dir, f"{prefix}_specificity.pkl")
        with open(output_path, "wb") as handle:
            pickle.dump(specificity_results, handle)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
