
import os
import csv
import numpy as np
from collections import defaultdict

KNOWN_FILE = "evaluation_results/verification_embeddings.npz"
UNKNOWN_FILE = "evaluation_results/unknown_embeddings.npz"

OUTPUT_DIR = "evaluation_results/gallery_evaluation"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "gallery_size_results.csv")

THRESHOLD = 0.32

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

for person in person_images:
    person_images[person].sort(key=lambda x: x[0])

unknown_embeddings = unknown["embeddings"]
unknown_names = unknown["image_names"]

results = []

print("=" * 75)
print("GALLERY SIZE EVALUATION")
print("=" * 75)

print(f"\nThreshold: {THRESHOLD}")
print("\n" + "-" * 75)
print(
    f"{'Gallery':<12}"
    f"{'Known correct':<18}"
    f"{'Known total':<15}"
    f"{'Unknown rejected':<20}"
    f"{'Unknown total':<15}"
)
print("-" * 75)


# Test gallery sizes from 1 to 5
for gallery_size in range(1, 6):

    gallery = {}

    for person in sorted(person_images):

        images = person_images[person]

        reference = images[:gallery_size]

        representative = normalize(
            np.mean([item[1] for item in reference], axis=0)
        )

        gallery[person] = representative

    known_correct = 0
    known_total = 0
    unknown_rejected = 0
    unknown_total = 0
    known_rejected = 0
    unknown_accepted = 0

    # Test registered participants
    for person in sorted(person_images):

        query_images = person_images[person][gallery_size:]

        for name, embedding in query_images:

            scores = {
                candidate: cosine_similarity(embedding, ref)
                for candidate, ref in gallery.items()
            }

            best_match = max(scores, key=scores.get)
            best_score = scores[best_match]

            prediction = (
                best_match if best_score >= THRESHOLD else "Unknown"
            )

            known_total += 1

            if prediction == person:
                known_correct += 1

            if prediction == "Unknown":
                known_rejected += 1

    # Test unknown participant P03
    for name, embedding in zip(unknown_names, unknown_embeddings):

        scores = {
            candidate: cosine_similarity(embedding, ref)
            for candidate, ref in gallery.items()
        }

        best_score = max(scores.values())

        prediction = (
            max(scores, key=scores.get)
            if best_score >= THRESHOLD
            else "Unknown"
        )

        unknown_total += 1

        if prediction == "Unknown":
            unknown_rejected += 1
        else:
            unknown_accepted += 1

    known_accuracy = (
        known_correct / known_total if known_total else 0
    )

    unknown_rejection_rate = (
        unknown_rejected / unknown_total if unknown_total else 0
    )

    results.append({
        "gallery_size": gallery_size,
        "known_correct": known_correct,
        "known_total": known_total,
        "known_rejected": known_rejected,
        "known_accuracy": known_accuracy,
        "unknown_rejected": unknown_rejected,
        "unknown_total": unknown_total,
        "unknown_accepted": unknown_accepted,
        "unknown_rejection_rate": unknown_rejection_rate
    })

    print(
        f"{gallery_size:<12}"
        f"{known_correct:<18}"
        f"{known_total:<15}"
        f"{unknown_rejected:<20}"
        f"{unknown_total:<15}"
    )


# Save CSV
with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


print("\n" + "=" * 75)
print("GALLERY SIZE EVALUATION COMPLETED")
print("=" * 75)

print(f"\n[OUTPUT] {OUTPUT_CSV}")
