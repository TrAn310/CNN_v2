
import os
import csv
import numpy as np
from collections import defaultdict

# =====================================================
# CONFIGURATION
# =====================================================

KNOWN_FILE = "evaluation_results/verification_embeddings.npz"
UNKNOWN_FILE = "evaluation_results/unknown_embeddings.npz"

OUTPUT_DIR = "evaluation_results/unknown_evaluation"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "unknown_results.csv")

GALLERY_SIZE = 4
THRESHOLD = 0.32

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def normalize(vector):
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Zero-norm embedding.")

    return vector / norm


def cosine_similarity(a, b):
    return float(np.dot(normalize(a), normalize(b)))


# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("UNKNOWN PERSON REJECTION EVALUATION")
print("=" * 60)

known_data = np.load(KNOWN_FILE)
unknown_data = np.load(UNKNOWN_FILE)

known_embeddings = known_data["embeddings"]
known_participants = known_data["participants"]
known_names = known_data["image_names"]

unknown_embeddings = unknown_data["embeddings"]
unknown_names = unknown_data["image_names"]

print("\n[INFO] Known images:", len(known_embeddings))
print("[INFO] Unknown images:", len(unknown_embeddings))


# =====================================================
# BUILD GALLERY FROM P01 AND P02
# =====================================================

person_images = defaultdict(list)

for embedding, person, name in zip(
    known_embeddings,
    known_participants,
    known_names
):
    person_images[str(person)].append((str(name), embedding))

gallery = {}

print("\n[GALLERY]")

for person in sorted(person_images):

    images = sorted(
        person_images[person],
        key=lambda item: item[0]
    )

    if len(images) <= GALLERY_SIZE:
        raise ValueError(f"Not enough images for {person}")

    reference_images = images[:GALLERY_SIZE]

    representative = normalize(
        np.mean(
            [item[1] for item in reference_images],
            axis=0
        )
    )

    gallery[person] = representative

    print(f"{person}: {len(reference_images)} reference images")


# =====================================================
# BUILD QUERY SET
# =====================================================

queries = []

# Known participants: use images 5-7
for person in sorted(person_images):

    images = sorted(
        person_images[person],
        key=lambda item: item[0]
    )

    for name, embedding in images[GALLERY_SIZE:]:

        queries.append({
            "image": name,
            "actual": person,
            "embedding": embedding
        })

# Unknown participant: use all P03 images
for name, embedding in zip(unknown_names, unknown_embeddings):

    queries.append({
        "image": str(name),
        "actual": "Unknown",
        "embedding": embedding
    })


# =====================================================
# IDENTIFICATION AND UNKNOWN REJECTION
# =====================================================

results = []

known_correct = 0
known_total = 0

unknown_correct = 0
unknown_total = 0

unknown_false_accepts = 0

print("\n" + "=" * 60)
print("PREDICTION RESULTS")
print("=" * 60)

for query in queries:

    scores = {}

    for person, representative in gallery.items():

        scores[person] = cosine_similarity(
            query["embedding"],
            representative
        )

    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]

    if best_score >= THRESHOLD:
        prediction = best_match
    else:
        prediction = "Unknown"

    correct = prediction == query["actual"]

    if query["actual"] == "Unknown":

        unknown_total += 1

        if prediction == "Unknown":
            unknown_correct += 1
        else:
            unknown_false_accepts += 1

    else:

        known_total += 1

        if correct:
            known_correct += 1

    result = {
        "image": query["image"],
        "actual": query["actual"],
        "predicted": prediction,
        "best_match": best_match,
        "similarity": best_score,
        "correct": correct
    }

    results.append(result)

    print(f"\nImage: {query['image']}")
    print(f"Actual: {query['actual']}")
    print(f"Predicted: {prediction}")
    print(f"Best match: {best_match}")
    print(f"Similarity: {best_score:.6f}")
    print(f"Result: {'CORRECT' if correct else 'INCORRECT'}")


# =====================================================
# SUMMARY
# =====================================================

known_accuracy = (
    known_correct / known_total if known_total else 0
)

unknown_rejection_rate = (
    unknown_correct / unknown_total if unknown_total else 0
)

overall_accuracy = (
    sum(r["correct"] for r in results) / len(results)
    if results else 0
)

print("\n" + "=" * 60)
print("UNKNOWN REJECTION EVALUATION REPORT")
print("=" * 60)

print("\n[KNOWN PERSONS]")
print(f"Total queries: {known_total}")
print(f"Correct identifications: {known_correct}")
print(f"Identification accuracy: {known_accuracy * 100:.2f}%")

print("\n[UNKNOWN PERSON]")
print(f"Total queries: {unknown_total}")
print(f"Correctly rejected: {unknown_correct}")
print(f"False acceptances: {unknown_false_accepts}")
print(f"Unknown rejection rate: {unknown_rejection_rate * 100:.2f}%")

print("\n[OVERALL]")
print(f"Total queries: {len(results)}")
print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")
print(f"Threshold: {THRESHOLD:.2f}")


# =====================================================
# SAVE CSV
# =====================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "actual",
            "predicted",
            "best_match",
            "similarity",
            "correct"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\n[OUTPUT] CSV: {OUTPUT_CSV}")

print("\nEVALUATION COMPLETED")
