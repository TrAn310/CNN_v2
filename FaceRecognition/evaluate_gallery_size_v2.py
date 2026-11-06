
import os
import csv
import numpy as np
from collections import defaultdict

KNOWN_FILE = "evaluation_results/verification_embeddings.npz"
UNKNOWN_FILE = "evaluation_results/unknown_embeddings.npz"

OUTPUT_DIR = "evaluation_results/gallery_evaluation"
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "gallery_size_results_v2.csv"
)

THRESHOLD = 0.32
QUERY_START = 4  # Images 5, 6, 7

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
print("GALLERY SIZE EVALUATION - FIXED QUERY SET")
print("=" * 75)

print(f"\nThreshold: {THRESHOLD}")
print("Known queries: images 5, 6, 7 of each participant")
print("Unknown queries: all P03 images")

for gallery_size in range(1, 5):

    gallery = {}

    # Build gallery using only the first N images
    for person in sorted(person_images):

        images = person_images[person]
        reference = images[:gallery_size]

        representative = normalize(
            np.mean([item[1] for item in reference], axis=0)
        )

        gallery[person] = representative

    known_correct = 0
    known_total = 0
    known_rejected = 0
    known_misidentified = 0

    unknown_rejected = 0
    unknown_accepted = 0
    unknown_total = 0

    # Fixed known query set: images 5, 6, 7
    for person in sorted(person_images):

        query_images = person_images[person][QUERY_START:]

        for name, embedding in query_images:

            scores = {
                candidate: cosine_similarity(embedding, ref)
                for candidate, ref in gallery.items()
            }

            best_match = max(scores, key=scores.get)
            best_score = scores[best_match]

            if best_score < THRESHOLD:
                prediction = "Unknown"
                known_rejected += 1
            else:
                prediction = best_match

                if prediction != person:
                    known_misidentified += 1

            if prediction == person:
                known_correct += 1

            known_total += 1

    # Fixed unknown query set: all P03 images
    for name, embedding in zip(unknown_names, unknown_embeddings):

        scores = {
            candidate: cosine_similarity(embedding, ref)
            for candidate, ref in gallery.items()
        }

        best_match = max(scores, key=scores.get)
        best_score = scores[best_match]

        if best_score < THRESHOLD:
            unknown_rejected += 1
        else:
            unknown_accepted += 1

        unknown_total += 1

    known_accuracy = known_correct / known_total
    unknown_rejection_rate = unknown_rejected / unknown_total

    results.append({
        "gallery_size": gallery_size,
        "known_correct": known_correct,
        "known_total": known_total,
        "known_rejected": known_rejected,
        "known_misidentified": known_misidentified,
        "known_accuracy": known_accuracy,
        "unknown_rejected": unknown_rejected,
        "unknown_accepted": unknown_accepted,
        "unknown_total": unknown_total,
        "unknown_rejection_rate": unknown_rejection_rate
    })

    print(f"\nGallery size: {gallery_size}")
    print(f"Known identification: {known_correct}/{known_total}")
    print(f"Known rejected: {known_rejected}")
    print(f"Known misidentified: {known_misidentified}")
    print(f"Unknown rejected: {unknown_rejected}/{unknown_total}")
    print(f"Unknown accepted: {unknown_accepted}")


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
print("FIXED QUERY EVALUATION COMPLETED")
print("=" * 75)

print(f"[OUTPUT] {OUTPUT_CSV}")
