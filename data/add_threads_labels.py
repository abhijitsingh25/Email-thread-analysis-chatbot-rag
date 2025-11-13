import pandas as pd
import json

# Load your EXISTING sliced CSV
df = pd.read_csv("data/sample_enron_slice.csv")

# Generate sorted unique thread IDs
unique_threads = sorted(df["thread_id"].unique())

# Create mapping
thread_map = {tid: f"T-{i+1:04d}" for i, tid in enumerate(unique_threads)}

# Add new column
df["thread_label"] = df["thread_id"].map(thread_map)

# Save updated CSV
df.to_csv("data/sliced_emails_labeled.csv", index=False)

# Save mapping JSON for backend/Streamlit
with open("data/thread_map.json", "w") as f:
    json.dump(thread_map, f, indent=4)

print("✅ Added thread_label column")
print("✅ Created thread_map.json")
print(f"Total threads: {len(unique_threads)}")
