
from pathlib import Path
from itertools import combinations

import numpy as np
import csv


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "evaluation_results"
    / "verification_embeddings.npz"
)

OUTPUT_DIR = BASE_DIR / "evaluation_results"
OUTPUT_FILE = OUTPUT_DIR / "similarity_report.csv"


# =====================================
# 2. LOAD EMBEDDINGS
# =====================================

data = np.load(INPUT_FILE)

embeddings = data["embeddings"]
participants = data["participants"]
image_names = data["image_names"]

print("[INFO] Loaded embeddings:", embeddings.shape)


# =====================================
# 3. COSINE SIMILARITY
# =====================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# =====================================
# 4. GENERATE PAIRS
# =====================================

pairs = []

for i, j in combinations(range(len(embeddings)), 2):

    score = cosine_similarity(
        embeddings[i],
        embeddings[j]
    )

    same_person = participants[i] == participants[j]

    pairs.append({
        "image_1": image_names[i],
        "image_2": image_names[j],
        "participant_1": participants[i],
        "participant_2": participants[j],
        "pair_type": (
            "SAME" if same_person else "DIFFERENT"
        ),
        "cosine_similarity": score
    })


# =====================================
# 5. STATISTICS
# =====================================

same_scores = [
    p["cosine_similarity"]
    for p in pairs
    if p["pair_type"] == "SAME"
]

different_scores = [
    p["cosine_similarity"]
    for p in pairs
    if p["pair_type"] == "DIFFERENT"
]


# =====================================
# 6. SAVE CSV
# =====================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

columns = [
    "image_1",
    "image_2",
    "participant_1",
    "participant_2",
    "pair_type",
    "cosine_similarity"
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
    writer.writerows(pairs)


# =====================================
# 7. DISPLAY RESULTS
# =====================================

print("\n" + "=" * 50)
print("COSINE SIMILARITY EVALUATION")
print("=" * 50)

print("Total pairs:", len(pairs))

print("\nSAME PERSON:")
print("Number of pairs:", len(same_scores))
print("Mean:", np.mean(same_scores))
print("Min:", np.min(same_scores))
print("Max:", np.max(same_scores))

print("\nDIFFERENT PERSON:")
print("Number of pairs:", len(different_scores))
print("Mean:", np.mean(different_scores))
print("Min:", np.min(different_scores))
print("Max:", np.max(different_scores))

print("\nCSV saved to:", OUTPUT_FILE)
