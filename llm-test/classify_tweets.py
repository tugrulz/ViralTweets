"""
LLM-based virality classifier using Llama-3.3-70B via ELM API.

Loads text-only tweets from the ViralTweets dataset, asks the LLM to
classify each tweet as VIRAL or NON-VIRAL, then evaluates against the
ground-truth `viral` column.

Usage:
    python classify_tweets.py [--n-per-class N] [--seed S] [--output-dir DIR]

Defaults: 150 samples per class, seed=42, output saved to results/.
"""

import argparse
import json
import os
import time
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
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.environ["ELM_API_KEY"]
BASE_URL = os.environ.get("ELM_BASE_URL", "https://elm.edina.ac.uk/api/v1")
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

DATA_PATH = Path(__file__).parent.parent / (
    "classification/model_with_extra_features/"
    "final_dataset_since_october_2022.parquet.gzip"
)

# ---------------------------------------------------------------------------
# Prompt
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
# Helpers
# ---------------------------------------------------------------------------


def load_text_only_tweets(data_path: Path) -> pd.DataFrame:
    """Load dataset, keep English text-only tweets, return relevant columns."""
    print(f"Loading data from {data_path} …")
    df = pd.read_parquet(data_path)
    print(f"  Total tweets loaded : {len(df):,}")

    # Keep English text-only tweets
    mask = (df["has_media"] == False) & (df["lang"] == "en")  # noqa: E712
    df = df.loc[mask, ["id", "text", "viral", "followers_count", "retweet_count"]].copy()
    print(f"  English, text-only  : {len(df):,}")
    print(f"  Viral               : {df['viral'].sum():,}  ({df['viral'].mean()*100:.2f}%)")
    return df


def sample_balanced(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """Return a balanced sample: n_per_class viral + n_per_class non-viral."""
    viral = df[df["viral"] == True]  # noqa: E712
    non_viral = df[df["viral"] == False]  # noqa: E712

    n_viral = min(n_per_class, len(viral))
    n_non_viral = min(n_per_class, len(non_viral))

    sample_viral = viral.sample(n=n_viral, random_state=seed)
    sample_non_viral = non_viral.sample(n=n_non_viral, random_state=seed)

    sample = pd.concat([sample_viral, sample_non_viral]).sample(frac=1, random_state=seed)
    print(f"\nSampled {n_viral} viral + {n_non_viral} non-viral = {len(sample)} tweets total")
    return sample.reset_index(drop=True)


def call_llm(client: OpenAI, text: str, retries: int = 3) -> tuple[str, str]:
    """
    Ask the LLM to classify a single tweet.

    Returns (label, raw_response) where label is 'VIRAL' or 'NON-VIRAL'.
    Falls back to 'UNKNOWN' if parsing fails after all retries.
    """
    for attempt in range(1, retries + 1):
        try:
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
            raw = response.choices[0].message.content.strip().upper()
            if "NON-VIRAL" in raw:
                return "NON-VIRAL", raw
            elif "VIRAL" in raw:
                return "VIRAL", raw
            # Unexpected output — retry
            print(f"    Unexpected response: '{raw}' (attempt {attempt})")
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    API error on attempt {attempt}: {exc}  (retrying in {wait}s)")
            time.sleep(wait)

    return "UNKNOWN", "FAILED"


def run_classification(
    df: pd.DataFrame,
    client: OpenAI,
    delay: float = 0.3,
) -> pd.DataFrame:
    """Classify all tweets in df; returns df with new columns."""
    predictions, raw_responses = [], []

    for i, row in df.iterrows():
        label, raw = call_llm(client, row["text"])
        predictions.append(label)
        raw_responses.append(raw)

        done = len(predictions)
        if done % 10 == 0 or done == len(df):
            print(f"  [{done}/{len(df)}] last='{label}'")

        time.sleep(delay)

    df = df.copy()
    df["llm_prediction"] = predictions
    df["llm_raw"] = raw_responses
    return df


def evaluate(df: pd.DataFrame) -> dict:
    """Compute classification metrics; skip UNKNOWN predictions."""
    df_eval = df[df["llm_prediction"] != "UNKNOWN"].copy()
    skipped = len(df) - len(df_eval)
    if skipped:
        print(f"\nWarning: {skipped} tweets had UNKNOWN predictions and are excluded from metrics.")

    y_true = df_eval["viral"].astype(int)
    y_pred = (df_eval["llm_prediction"] == "VIRAL").astype(int)

    metrics = {
        "n_evaluated": len(df_eval),
        "n_skipped": skipped,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_viral": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_viral": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_viral": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["non-viral", "viral"], zero_division=0
        ),
    }

    # Baseline: predict everything as non-viral (majority class)
    y_majority = [0] * len(y_true)
    metrics["baseline_accuracy"] = round(accuracy_score(y_true, y_majority), 4)
    metrics["baseline_f1_viral"] = round(f1_score(y_true, y_majority, zero_division=0), 4)

    return metrics


def print_results(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("LLAMA VIRALITY CLASSIFICATION — RESULTS")
    print("=" * 60)
    print(f"Model            : {MODEL}")
    print(f"Tweets evaluated : {metrics['n_evaluated']}")
    print()
    print(f"{'Metric':<25} {'LLM':>8}   {'Baseline (all-non-viral)':>24}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {metrics['accuracy']:>8.4f}   {metrics['baseline_accuracy']:>24.4f}")
    print(f"{'F1 (viral class)':<25} {metrics['f1_viral']:>8.4f}   {metrics['baseline_f1_viral']:>24.4f}")
    print(f"{'Precision (viral)':<25} {metrics['precision_viral']:>8.4f}")
    print(f"{'Recall (viral)':<25} {metrics['recall_viral']:>8.4f}")
    print()
    print("Confusion Matrix (rows=true, cols=pred):")
    print("                 Pred NON-VIRAL  Pred VIRAL")
    cm = metrics["confusion_matrix"]
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
    parser.add_argument("--n-per-class", type=int, default=150,
                        help="Number of tweets per class to classify (default: 150)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds to wait between API calls (default: 0.3)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load & sample
    df = load_text_only_tweets(DATA_PATH)
    sample = sample_balanced(df, n_per_class=args.n_per_class, seed=args.seed)

    # Print a few examples
    print("\n--- Sample tweets (first 3 viral) ---")
    for _, row in sample[sample["viral"] == True].head(3).iterrows():  # noqa: E712
        print(f"  [{row['retweet_count']} RTs] {row['text'][:120]}")

    # Run LLM classification
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"\nClassifying {len(sample)} tweets with {MODEL} …")
    results_df = run_classification(sample, client, delay=args.delay)

    # Save per-tweet results
    out_csv = args.output_dir / "predictions.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nPer-tweet predictions saved to: {out_csv}")

    # Evaluate
    metrics = evaluate(results_df)
    print_results(metrics)

    # Save metrics JSON
    out_json = args.output_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
