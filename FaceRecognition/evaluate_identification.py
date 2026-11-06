
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "evaluation_results/verification_embeddings.npz"
OUTPUT_DIR = "evaluation_results/identification"

# Ngưỡng thử nghiệm ban đầu, chưa được hiệu chuẩn riêng
THRESHOLD = 0.32

# Số ảnh dùng để đăng ký mỗi người
GALLERY_SIZE = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(vector):
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Embedding has zero norm.")

    return vector / norm


def cosine_similarity(a, b):
    return float(np.dot(normalize(a), normalize(b)))


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STARTING FACE IDENTIFICATION EVALUATION")
print("=" * 60)

data = np.load(INPUT_FILE)

embeddings = data["embeddings"]
participants = data["participants"]
image_names = data["image_names"]

print("\n[INFO] Data loaded successfully.")
print(f"[INFO] Number of images: {len(embeddings)}")
print(f"[INFO] Embedding shape: {embeddings.shape}")


# ============================================================
# SPLIT GALLERY AND QUERY
# ============================================================

person_images = defaultdict(list)

for embedding, participant, image_name in zip(
    embeddings, participants, image_names
):
    person_images[str(participant)].append(
        (str(image_name), embedding)
    )

gallery = {}
queries = []

print("\n[GALLERY / QUERY SPLIT]")

for person in sorted(person_images.keys()):

    images = sorted(
        person_images[person],
        key=lambda item: item[0]
    )

    if len(images) <= GALLERY_SIZE:
        raise ValueError(
            f"Not enough images for participant {person}"
        )

    reference_images = images[:GALLERY_SIZE]
    query_images = images[GALLERY_SIZE:]

    # Tạo embedding đại diện bằng trung bình các embedding
    reference_embeddings = np.array(
        [item[1] for item in reference_images]
    )

    representative = normalize(
        np.mean(reference_embeddings, axis=0)
    )

    gallery[person] = representative

    print(
        f"{person}: "
        f"{len(reference_images)} gallery images, "
        f"{len(query_images)} query images"
    )

    for image_name, embedding in query_images:
        queries.append({
            "image": image_name,
            "actual": person,
            "embedding": embedding
        })


# ============================================================
# IDENTIFICATION
# ============================================================

print("\n" + "=" * 60)
print("IDENTIFICATION RESULTS")
print("=" * 60)

results = []

correct = 0
rejected = 0

for query in queries:

    scores = {}

    for person, representative in gallery.items():
        scores[person] = cosine_similarity(
            query["embedding"],
            representative
        )

    predicted_person = max(scores, key=scores.get)
    best_score = scores[predicted_person]

    if best_score >= THRESHOLD:
        final_prediction = predicted_person
    else:
        final_prediction = "Unknown"
        rejected += 1

    is_correct = final_prediction == query["actual"]

    if is_correct:
        correct += 1

    results.append({
        "image": query["image"],
        "actual": query["actual"],
        "predicted": final_prediction,
        "best_match": predicted_person,
        "similarity": best_score,
        "correct": is_correct
    })

    print(f"\nImage: {query['image']}")
    print(f"Actual person: {query['actual']}")
    print(f"Predicted: {final_prediction}")
    print(f"Best match: {predicted_person}")
    print(f"Similarity: {best_score:.6f}")
    print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")


# ============================================================
# EVALUATION SUMMARY
# ============================================================

total = len(results)
accuracy = correct / total if total else 0

print("\n" + "=" * 60)
print("FACE IDENTIFICATION - EVALUATION REPORT")
print("=" * 60)

print(f"\nNumber of participants: {len(gallery)}")
print(f"Gallery images per participant: {GALLERY_SIZE}")
print(f"Total query images: {total}")
print(f"Correct identifications: {correct}")
print(f"Incorrect / rejected: {total - correct}")
print(f"Unknown rejections: {rejected}")
print(f"Threshold: {THRESHOLD:.2f}")
print(f"Identification accuracy: {accuracy * 100:.2f}%")


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "identification_results.csv"
)

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:

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

print(f"\n[OUTPUT] CSV: {csv_path}")


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 5))

labels = [item["image"] for item in results]
values = [item["similarity"] for item in results]

bars = plt.bar(labels, values)

plt.axhline(
    y=THRESHOLD,
    linestyle="--",
    label=f"Threshold = {THRESHOLD}"
)

plt.ylim(0, 1)
plt.ylabel("Cosine Similarity")
plt.xlabel("Query Images")
plt.title("Face Identification Similarity Scores")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

chart_path = os.path.join(
    OUTPUT_DIR,
    "identification_similarity.png"
)

plt.savefig(chart_path, dpi=300)
plt.close()

print(f"[OUTPUT] Chart: {chart_path}")

print("\n" + "=" * 60)
print("IDENTIFICATION EVALUATION COMPLETED")
print("=" * 60)
