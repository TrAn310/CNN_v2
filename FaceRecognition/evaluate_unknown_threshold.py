
import os
import csv
import numpy as np
from collections import defaultdict

KNOWN_FILE = "evaluation_results/verification_embeddings.npz"
UNKNOWN_FILE = "evaluation_results/unknown_embeddings.npz"

OUTPUT_DIR = "evaluation_results/unknown_evaluation"
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "threshold_sweep_results.csv"
)

GALLERY_SIZE = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        raise ValueError("Zero norm embedding")
    return x / norm


def cosine_similarity(a, b):
    return float(np.dot(normalize(a), normalize(b)))


# Load data
known = np.load(KNOWN_FILE)
unknown = np.load(UNKNOWN_FILE)

person_images = defaultdict(list)

for emb, person, name in zip(
    known["embeddings"],
    known["participants"],
    known["image_names"]
):
    person_images[str(person)].append((str(name), emb))


# Build gallery
gallery = {}

for person in sorted(person_images):

    images = sorted(person_images[person], key=lambda x: x[0])

    reference = images[:GALLERY_SIZE]

    representative = normalize(
        np.mean([x[1] for x in reference], axis=0)
    )

    gallery[person] = representative


# Build queries
queries = []

for person in sorted(person_images):

    images = sorted(person_images[person], key=lambda x: x[0])

    for name, emb in images[GALLERY_SIZE:]:
        queries.append({
            "actual": person,
            "embedding": emb
        })


for name, emb in zip(
    unknown["image_names"],
    unknown["embeddings"]
):
    queries.append({
        "actual": "Unknown",
        "embedding": emb
    })


# Calculate best similarity for each query
for query in queries:

    scores = {
        person: cosine_similarity(query["embedding"], ref)
        for person, ref in gallery.items()
    }

    query["best_match"] = max(scores, key=scores.get)
    query["best_score"] = scores[query["best_match"]]


# Sweep thresholds
thresholds = np.round(np.arange(0.10, 0.801, 0.01), 2)

results = []

for threshold in thresholds:

    known_total = 0
    known_correct = 0
    known_rejected = 0
    known_misidentified = 0

    unknown_total = 0
    unknown_rejected = 0
    unknown_accepted = 0

    for query in queries:

        actual = query["actual"]
        score = query["best_score"]
        best_match = query["best_match"]

        if actual == "Unknown":

            unknown_total += 1

            if score >= threshold:
                unknown_accepted += 1
            else:
                unknown_rejected += 1

        else:

            known_total += 1

            if score < threshold:
                known_rejected += 1

            elif best_match == actual:
                known_correct += 1

            else:
                known_misidentified += 1

    known_accuracy = known_correct / known_total
    unknown_far = unknown_accepted / unknown_total
    unknown_rejection = unknown_rejected / unknown_total

    results.append({
        "threshold": threshold,
        "known_correct": known_correct,
        "known_rejected": known_rejected,
        "known_misidentified": known_misidentified,
        "known_accuracy": known_accuracy,
        "unknown_accepted": unknown_accepted,
        "unknown_rejected": unknown_rejected,
        "unknown_far": unknown_far,
        "unknown_rejection_rate": unknown_rejection
    })


# Print selected thresholds
print("=" * 75)
print("UNKNOWN REJECTION - THRESHOLD ANALYSIS")
print("=" * 75)

print(
    f"\n{'Threshold':<12}"
    f"{'Known correct':<16}"
    f"{'Known rejected':<17}"
    f"{'Unknown FAR':<15}"
    f"{'Unknown reject':<15}"
)

for r in results:

    if r["threshold"] in [0.15, 0.20, 0.23, 0.25, 0.30, 0.32,
                          0.40, 0.45, 0.50, 0.60]:

        print(
            f"{r['threshold']:<12.2f}"
            f"{r['known_correct']:<16}"
            f"{r['known_rejected']:<17}"
            f"{r['unknown_far'] * 100:<15.2f}"
            f"{r['unknown_rejection_rate'] * 100:<15.2f}"
        )


# Save CSV
with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# Find thresholds satisfying zero errors on this sample
valid = [
    r for r in results
    if r["known_correct"] == 6
    and r["known_rejected"] == 0
    and r["known_misidentified"] == 0
    and r["unknown_accepted"] == 0
]

print("\n" + "=" * 75)
print("ZERO-ERROR THRESHOLD RANGE ON CURRENT TEST DATA")
print("=" * 75)

if valid:
    print(
        f"From {valid[0]['threshold']:.2f} "
        f"to {valid[-1]['threshold']:.2f}"
    )
else:
    print("No threshold satisfies all conditions.")

print(f"\n[OUTPUT] {OUTPUT_CSV}")
print("\nTHRESHOLD ANALYSIS COMPLETED")
