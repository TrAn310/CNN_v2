
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "evaluation_results"

SIMILARITY_FILE = RESULT_DIR / "similarity_report.csv"
THRESHOLD_FILE = RESULT_DIR / "threshold_report.csv"

OUTPUT_DIR = RESULT_DIR / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_THRESHOLD = 0.32


# =====================================
# 2. LOAD SIMILARITY DATA
# =====================================

same_scores = []
different_scores = []

with SIMILARITY_FILE.open(
    "r", encoding="utf-8-sig", newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        score = float(row["cosine_similarity"])

        if row["pair_type"] == "SAME":
            same_scores.append(score)

        elif row["pair_type"] == "DIFFERENT":
            different_scores.append(score)


# =====================================
# 3. LOAD THRESHOLD DATA
# =====================================

thresholds = []
far_values = []
frr_values = []
accuracy_values = []

with THRESHOLD_FILE.open(
    "r", encoding="utf-8-sig", newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        thresholds.append(float(row["threshold"]))
        far_values.append(float(row["FAR"]) * 100)
        frr_values.append(float(row["FRR"]) * 100)
        accuracy_values.append(float(row["Accuracy"]) * 100)


thresholds = np.array(thresholds)
far_values = np.array(far_values)
frr_values = np.array(frr_values)
accuracy_values = np.array(accuracy_values)


# =====================================
# 4. CHART 1: SIMILARITY DISTRIBUTION
# =====================================

plt.figure(figsize=(9, 5))

plt.hist(
    same_scores,
    bins=12,
    alpha=0.6,
    label="Same person"
)

plt.hist(
    different_scores,
    bins=12,
    alpha=0.6,
    label="Different person"
)

plt.axvline(
    SELECTED_THRESHOLD,
    linestyle="--",
    label=f"Threshold = {SELECTED_THRESHOLD}"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Number of pairs")
plt.title("Cosine Similarity Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_similarity_distribution.png", dpi=300)
plt.close()


# =====================================
# 5. CHART 2: FAR
# =====================================

plt.figure(figsize=(9, 5))

plt.plot(thresholds, far_values)

plt.axvline(
    SELECTED_THRESHOLD,
    linestyle="--",
    label=f"Threshold = {SELECTED_THRESHOLD}"
)

plt.xlabel("Threshold")
plt.ylabel("FAR (%)")
plt.title("False Acceptance Rate (FAR)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_far_curve.png", dpi=300)
plt.close()


# =====================================
# 6. CHART 3: FRR
# =====================================

plt.figure(figsize=(9, 5))

plt.plot(thresholds, frr_values)

plt.axvline(
    SELECTED_THRESHOLD,
    linestyle="--",
    label=f"Threshold = {SELECTED_THRESHOLD}"
)

plt.xlabel("Threshold")
plt.ylabel("FRR (%)")
plt.title("False Rejection Rate (FRR)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_frr_curve.png", dpi=300)
plt.close()


# =====================================
# 7. CHART 4: FAR AND FRR
# =====================================

valid_indices = np.where(
    (far_values == 0) & (frr_values == 0)
)[0]

plt.figure(figsize=(9, 5))

plt.plot(thresholds, far_values, label="FAR")
plt.plot(thresholds, frr_values, label="FRR")

plt.axvline(
    SELECTED_THRESHOLD,
    linestyle="--",
    label=f"Selected threshold = {SELECTED_THRESHOLD}"
)

if len(valid_indices) > 0:
    min_t = thresholds[valid_indices[0]]
    max_t = thresholds[valid_indices[-1]]

    plt.axvspan(
        min_t,
        max_t,
        alpha=0.2,
        label=f"Zero-error interval: {min_t:.2f}-{max_t:.2f}"
    )

plt.xlabel("Threshold")
plt.ylabel("Error Rate (%)")
plt.title("FAR and FRR Comparison")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_far_frr_comparison.png", dpi=300)
plt.close()


# =====================================
# 8. SUMMARY
# =====================================

print("=" * 50)
print("VISUALIZATION COMPLETED")
print("=" * 50)

print("Selected threshold:", SELECTED_THRESHOLD)

if len(valid_indices) > 0:
    print(
        "Zero-error interval:",
        f"{min_t:.2f} - {max_t:.2f}"
    )
else:
    print("No zero-error interval found.")

print("\nGenerated charts:")

for path in sorted(OUTPUT_DIR.glob("*.png")):
    print(path)

print("\nDone.")
