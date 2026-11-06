
from pathlib import Path
import csv
import numpy as np


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "evaluation_results"
    / "similarity_report.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation_results"
    / "threshold_report.csv"
)


# =====================================
# 2. LOAD SIMILARITY DATA
# =====================================

same_scores = []
different_scores = []

with INPUT_FILE.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        score = float(row["cosine_similarity"])

        if row["pair_type"] == "SAME":
            same_scores.append(score)

        elif row["pair_type"] == "DIFFERENT":
            different_scores.append(score)


same_scores = np.array(same_scores)
different_scores = np.array(different_scores)

if len(same_scores) == 0 or len(different_scores) == 0:
    raise ValueError("Missing SAME or DIFFERENT pairs.")


# =====================================
# 3. EVALUATE THRESHOLDS
# =====================================

thresholds = np.arange(0.00, 1.001, 0.01)

results = []

for threshold in thresholds:

    # Same person predicted as same
    tp = int(np.sum(same_scores >= threshold))

    # Same person predicted as different
    fn = int(np.sum(same_scores < threshold))

    # Different person predicted as same
    fp = int(np.sum(different_scores >= threshold))

    # Different person predicted as different
    tn = int(np.sum(different_scores < threshold))

    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    results.append({
        "threshold": round(float(threshold), 2),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "FAR": far,
        "FRR": frr,
        "Accuracy": accuracy,
        "FAR_FRR_difference": abs(far - frr)
    })


# =====================================
# 4. FIND APPROXIMATE EER POINT
# =====================================

eer_result = min(
    results,
    key=lambda row: row["FAR_FRR_difference"]
)


# =====================================
# 5. SAVE REPORT
# =====================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

columns = [
    "threshold",
    "TP",
    "TN",
    "FP",
    "FN",
    "FAR",
    "FRR",
    "Accuracy",
    "FAR_FRR_difference"
]

with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns
    )

    writer.writeheader()
    writer.writerows(results)


# =====================================
# 6. DISPLAY RESULTS
# =====================================

print("\n" + "=" * 60)
print("THRESHOLD EVALUATION")
print("=" * 60)

print("Same-person pairs:", len(same_scores))
print("Different-person pairs:", len(different_scores))

print("\nAPPROXIMATE EER POINT")
print("Threshold:", eer_result["threshold"])
print("FAR: {:.2f}%".format(eer_result["FAR"] * 100))
print("FRR: {:.2f}%".format(eer_result["FRR"] * 100))
print("Accuracy: {:.2f}%".format(
    eer_result["Accuracy"] * 100
))

print("\nSELECTED THRESHOLDS")

for threshold in [0.20, 0.30, 0.32, 0.40, 0.50, 0.60]:

    row = next(
        item for item in results
        if item["threshold"] == threshold
    )

    print(
        f"T={threshold:.2f} | "
        f"FAR={row['FAR'] * 100:.2f}% | "
        f"FRR={row['FRR'] * 100:.2f}% | "
        f"Accuracy={row['Accuracy'] * 100:.2f}%"
    )

print("\nReport saved to:", OUTPUT_FILE)
print("Done.")
