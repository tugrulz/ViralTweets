"""
Full LLM virality classifier — processes ALL text-only English viral tweets
plus an equal-sized random sample of non-viral tweets.

Features over classify_tweets.py:
  • Uses every text-only viral tweet in the dataset (~382)
  • Concurrent API calls via ThreadPoolExecutor for speed
  • Checkpoint / resume: saves progress to results/checkpoint.csv every
    --save-every N calls so a crash doesn't lose work
  • Per-tweet token usage tracking
  • ETA display

Usage:
    python classify_tweets_full.py [--workers W] [--save-every N] [--seed S]

Defaults: 8 workers, save every 25 calls, seed=42.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

API_KEY = os.environ["ELM_API_KEY"]
BASE_URL = os.environ.get("ELM_BASE_URL", "https://elm.edina.ac.uk/api/v1")
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

DATA_PATH = Path(__file__).parent.parent / (
    "classification/model_with_extra_features/"
    "final_dataset_since_october_2022.parquet.gzip"
)

CHECKPOINT_FILE = Path("results/checkpoint.csv")

# ---------------------------------------------------------------------------
# Prompt  (same as classify_tweets.py — kept identical for fair comparison)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a social media analyst specialising in Twitter virality.

A tweet is **viral** when it spreads far beyond the author's usual audience —
gaining an unusually high number of retweets relative to the author's follower
count and typical engagement. Non-viral tweets are ordinary posts that stay
within the author's normal reach.

Your job: classify a tweet as VIRAL or NON-VIRAL based solely on its text.

Signals that correlate with virality:
• Strong emotion — humour, outrage, awe, empathy, or surprise
• Novelty — breaking news, an unexpected take, or a striking fact
• Broad appeal — relatable to people far outside a niche
• Controversy or debate — a clear opinion that invites a reaction
• Conciseness + punch — short, shareable phrasing
• Calls to action — "RT if you agree", questions that beg a reply

Signals that correlate with non-virality:
• Niche or personal content with limited broader relevance
• Promotional / marketing language
• Routine updates or mundane observations
• Lengthy walls of text with no clear hook

Respond with **exactly one line**: the label VIRAL or NON-VIRAL, nothing else.\
"""

USER_TEMPLATE = "Tweet: {text}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_text_only_tweets(data_path: Path) -> pd.DataFrame:
    print(f"Loading data from {data_path} …")
    df = pd.read_parquet(data_path)
    print(f"  Total tweets: {len(df):,}")

    mask = (df["has_media"] == False) & (df["lang"] == "en")  # noqa: E712
    df = df.loc[mask, ["id", "text", "viral", "followers_count", "retweet_count"]].copy()
    print(f"  English text-only: {len(df):,}")

    viral_count = df["viral"].sum()
    non_viral_count = (~df["viral"]).sum()
    print(f"  Viral:     {viral_count:,}  ({df['viral'].mean()*100:.2f}%)")
    print(f"  Non-viral: {non_viral_count:,}")
    return df


def build_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Take ALL viral tweets; match with equal-sized non-viral sample."""
    viral = df[df["viral"] == True]  # noqa: E712
    non_viral = df[df["viral"] == False]  # noqa: E712

    n = len(viral)
    sample_nv = non_viral.sample(n=n, random_state=seed)
    sample = pd.concat([viral, sample_nv]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nSample: {len(viral)} viral + {n} non-viral = {len(sample)} total tweets")
    return sample


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Return set of already-classified tweet IDs."""
    if checkpoint_path.exists():
        done = pd.read_csv(checkpoint_path)
        ids = set(done["id"].astype(str))
        print(f"Checkpoint found: {len(ids)} tweets already classified, resuming …")
        return ids
    return set()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def classify_one(
    client: OpenAI,
    tweet_id: str,
    text: str,
    ground_truth: bool,
    retries: int = 3,
) -> dict:
    """Classify a single tweet; returns a result dict."""
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(text=text)},
                ],
                temperature=0.0,
                max_tokens=10,
                stream=False,
            )
            elapsed = time.time() - t0
            raw = response.choices[0].message.content.strip().upper()
            tokens = response.usage.total_tokens if response.usage else None

            if "NON-VIRAL" in raw:
                label = "NON-VIRAL"
            elif "VIRAL" in raw:
                label = "VIRAL"
            else:
                label = "UNKNOWN"

            return {
                "id": tweet_id,
                "ground_truth": ground_truth,
                "llm_prediction": label,
                "llm_raw": raw,
                "tokens": tokens,
                "latency_s": round(elapsed, 2),
            }

        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [{tweet_id}] API error (attempt {attempt}): {exc}  → retry in {wait}s")
            time.sleep(wait)

    return {
        "id": tweet_id,
        "ground_truth": ground_truth,
        "llm_prediction": "UNKNOWN",
        "llm_raw": "FAILED",
        "tokens": None,
        "latency_s": None,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(results: list[dict]) -> dict:
    df = pd.DataFrame(results)
    df_eval = df[df["llm_prediction"] != "UNKNOWN"]
    skipped = len(df) - len(df_eval)
    if skipped:
        print(f"\nWarning: {skipped} UNKNOWN predictions excluded from metrics.")

    y_true = df_eval["ground_truth"].astype(int)
    y_pred = (df_eval["llm_prediction"] == "VIRAL").astype(int)

    y_majority = [0] * len(y_true)

    metrics = {
        "model": MODEL,
        "n_total": len(df),
        "n_evaluated": len(df_eval),
        "n_skipped": skipped,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_viral": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_viral": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_viral": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "baseline_accuracy": round(accuracy_score(y_true, y_majority), 4),
        "baseline_f1_viral": round(f1_score(y_true, y_majority, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["non-viral", "viral"], zero_division=0
        ),
        "avg_latency_s": round(df_eval["latency_s"].mean(), 2) if "latency_s" in df_eval else None,
        "total_tokens": int(df_eval["tokens"].sum()) if df_eval["tokens"].notna().any() else None,
    }
    return metrics


def print_results(metrics: dict) -> None:
    print("\n" + "=" * 65)
    print("LLAMA VIRALITY CLASSIFICATION — FULL RUN RESULTS")
    print("=" * 65)
    print(f"Model            : {metrics['model']}")
    print(f"Tweets evaluated : {metrics['n_evaluated']} / {metrics['n_total']}")
    if metrics["avg_latency_s"]:
        print(f"Avg latency      : {metrics['avg_latency_s']}s/tweet")
    if metrics["total_tokens"]:
        print(f"Total tokens     : {metrics['total_tokens']:,}")
    print()
    print(f"{'Metric':<25} {'LLM':>8}   {'Baseline':>10}")
    print("-" * 48)
    print(f"{'Accuracy':<25} {metrics['accuracy']:>8.4f}   {metrics['baseline_accuracy']:>10.4f}")
    print(f"{'F1 (viral)':<25} {metrics['f1_viral']:>8.4f}   {metrics['baseline_f1_viral']:>10.4f}")
    print(f"{'Precision (viral)':<25} {metrics['precision_viral']:>8.4f}")
    print(f"{'Recall (viral)':<25} {metrics['recall_viral']:>8.4f}")
    print()
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = metrics["confusion_matrix"]
    print("                 Pred NON-VIRAL  Pred VIRAL")
    print(f"  True NON-VIRAL      {cm[0][0]:>7}     {cm[0][1]:>7}")
    print(f"  True VIRAL          {cm[1][0]:>7}     {cm[1][1]:>7}")
    print()
    print("Full Classification Report:")
    print(metrics["classification_report"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent API workers (default: 8)")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save checkpoint every N completions (default: 25)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.csv"

    # Load data
    df = load_text_only_tweets(DATA_PATH)
    sample = build_sample(df, seed=args.seed)

    # Resume from checkpoint
    done_ids = load_checkpoint(checkpoint_path)
    todo = sample[~sample["id"].astype(str).isin(done_ids)]
    print(f"Tweets remaining: {len(todo)}")

    if todo.empty:
        print("All tweets already classified — loading checkpoint for evaluation.")
        results = pd.read_csv(checkpoint_path).to_dict("records")
    else:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        results: list[dict] = (
            pd.read_csv(checkpoint_path).to_dict("records") if done_ids else []
        )

        start = time.time()
        completed = len(done_ids)
        total = len(sample)
        pending_buffer: list[dict] = []

        print(f"\nClassifying with {args.workers} workers …\n")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(classify_one, client, str(row["id"]), row["text"], row["viral"]): i
                for i, row in todo.iterrows()
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pending_buffer.append(result)
                completed += 1

                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 1
                eta = (total - completed) / rate
                print(
                    f"  [{completed}/{total}]  "
                    f"pred={result['llm_prediction']:<10}  "
                    f"ETA {eta/60:.1f}min"
                )

                if len(pending_buffer) >= args.save_every:
                    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                    pending_buffer.clear()
                    print(f"  ✓ Checkpoint saved ({completed} done)")

        # Final save
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        print(f"\nAll predictions saved to {checkpoint_path}")

    # Evaluate
    metrics = evaluate(results)
    print_results(metrics)

    # Save metrics
    out_json = args.output_dir / "metrics_full.json"
    with open(out_json, "w") as f:
        serialisable = {
            k: v for k, v in metrics.items() if k != "classification_report"
        }
        json.dump(serialisable, f, indent=2)
    print(f"Metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
