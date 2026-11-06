
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_DIR = "unknown_dataset/P03"
OUTPUT_DIR = "evaluation_results"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "unknown_embeddings.npz"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# INITIALIZE MODEL
# ==========================================

print("=" * 60)
print("EXTRACTING UNKNOWN PERSON EMBEDDINGS")
print("=" * 60)

print("\n[INFO] Initializing InsightFace buffalo_l...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("[INFO] Model initialized successfully.")

# ==========================================
# PROCESS IMAGES
# ==========================================

embeddings = []
image_names = []

image_files = sorted(
    [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ],
    key=lambda x: int(os.path.splitext(x)[0])
    if os.path.splitext(x)[0].isdigit()
    else x
)

print(f"\n[INFO] Found {len(image_files)} images.")

for index, filename in enumerate(image_files, start=1):

    image_path = os.path.join(DATASET_DIR, filename)

    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Cannot read: {filename}")
        continue

    faces = app.get(image)

    print(f"\n[{index}/{len(image_files)}] {filename}")
    print(f"Detected faces: {len(faces)}")

    if len(faces) != 1:
        print("[WARNING] Expected exactly one face. Skipping.")
        continue

    embedding = faces[0].normed_embedding

    if embedding is None or len(embedding) != 512:
        print("[WARNING] Invalid embedding. Skipping.")
        continue

    embeddings.append(embedding)
    image_names.append(f"P03/{filename}")

    print(f"[SUCCESS] Embedding extracted: {embedding.shape}")

# ==========================================
# SAVE RESULTS
# ==========================================

if len(embeddings) == 0:
    raise RuntimeError("No valid embeddings extracted.")

np.savez_compressed(
    OUTPUT_FILE,
    embeddings=np.array(embeddings, dtype=np.float32),
    image_names=np.array(image_names)
)

print("\n" + "=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)

print(f"Total images: {len(image_files)}")
print(f"Successful embeddings: {len(embeddings)}")
print(f"Skipped images: {len(image_files) - len(embeddings)}")
print(f"Embedding shape: {np.array(embeddings).shape}")
print(f"Saved file: {OUTPUT_FILE}")

print("\nEXTRACTION COMPLETED")
