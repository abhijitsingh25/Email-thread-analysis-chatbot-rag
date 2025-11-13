import pandas as pd
import re, os, hashlib, argparse, json
from datetime import datetime
from email import message_from_string

def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()

def extract_metadata(raw_msg: str):
    """Parse minimal metadata from the raw email text."""
    try:
        msg = message_from_string(raw_msg)
        mid = msg.get("Message-ID") or sha1(raw_msg[:100])
        subj = msg.get("Subject", "")
        date_str = msg.get("Date", "")
        try:
            date = datetime.strptime(date_str[:25], "%a, %d %b %Y %H:%M:%S")
        except Exception:
            date = None
        in_reply_to = msg.get("In-Reply-To", "")
        refs = msg.get("References", "")
        thread_id = sha1((in_reply_to or refs or subj).strip().lower() or mid)
        body = msg.get_payload()
        if isinstance(body, list):
            body = " ".join([b.get_payload() for b in body if hasattr(b, "get_payload")])
        return {
            "message_id": mid,
            "thread_id": thread_id,
            "subject": subj,
            "date": date,
            "body": body,
        }
    except Exception:
        return None

def filter_date_range(df, start, end):
    mask = (df["date"].notnull()) & (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]

def add_thread_labels(subset_df, out_dir):
    """Add T-XXXX labels and save thread_map.json."""
    unique_threads = sorted(subset_df["thread_id"].unique())
    thread_map = {tid: f"T-{i+1:04d}" for i, tid in enumerate(unique_threads)}

    # Add new column
    subset_df["thread_label"] = subset_df["thread_id"].map(thread_map)

    # Save mapping
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "thread_map.json"), "w") as f:
        json.dump(thread_map, f, indent=4)

    print(f"🆕 Added thread_label column and saved thread_map.json")
    return subset_df


def make_enron_slice(csv_path, out_path="sample_enron_slice.csv", start_date=None, end_date=None, max_threads=20, max_msgs=300):
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    rows = []
    for i, row in df.iterrows():
        meta = extract_metadata(row["message"])
        if meta:
            rows.append(meta)
    meta_df = pd.DataFrame(rows)
    print(f"Parsed {len(meta_df)} messages")

    if start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        meta_df = filter_date_range(meta_df, start, end)
        print(f"🗓 Filtered by date range: {len(meta_df)} messages")

    # group by thread
    thread_groups = meta_df.groupby("thread_id")
    threads_sorted = sorted(thread_groups, key=lambda g: len(g[1]), reverse=True)
    selected_threads = threads_sorted[:max_threads]

    subset_rows = []
    for t_id, group in selected_threads:
        for _, r in group.head(max_msgs // max_threads).iterrows():
            subset_rows.append(r)

    subset_df = pd.DataFrame(subset_rows).drop_duplicates(subset="message_id")

    print(f" Selected {subset_df.thread_id.nunique()} threads and {len(subset_df)} messages")

    total_text_mb = subset_df["body"].astype(str).str.len().sum() / (1024 * 1024)
    print(f" Approx text size: {total_text_mb:.2f} MB")

    if total_text_mb > 100:
        subset_df = subset_df.sample(frac=100/total_text_mb, random_state=42)
        print(f" Trimmed to ~100MB: {len(subset_df)} messages")

    # Add thread labels + mapping file
    subset_df = add_thread_labels(subset_df, out_dir=os.path.dirname(out_path))

    subset_df.to_csv(out_path, index=False)
    print(f" Saved slice → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Make a small slice of Enron dataset")
    p.add_argument("--src", required=True, help="Path to Enron CSV file (with file,message columns)")
    p.add_argument("--out", default="sample_enron_slice.csv", help="Output CSV path")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = p.parse_args()

    make_enron_slice(args.src, args.out, args.start, args.end)
