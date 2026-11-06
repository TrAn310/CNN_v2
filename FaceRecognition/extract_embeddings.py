
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "verification_dataset"
OUTPUT_DIR = BASE_DIR / "evaluation_results"

OUTPUT_FILE = OUTPUT_DIR / "verification_embeddings.npz"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# =====================================
# 2. INITIALIZE INSIGHTFACE
# =====================================

print("[INFO] Initializing InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("[INFO] Model initialized successfully.\n")


# =====================================
# 3. EXTRACT EMBEDDINGS
# =====================================

embeddings = []
participants = []
image_names = []

errors = []

participant_dirs = sorted(
    p for p in DATASET_DIR.iterdir()
    if p.is_dir()
)

for participant_dir in participant_dirs:

    participant = participant_dir.name

    image_paths = sorted(
        p for p in participant_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"\n[INFO] Processing {participant}")

    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        if image is None:
            errors.append(f"{image_path}: Cannot read image")
            continue

        faces = app.get(image)

        # Require exactly one face per image
        if len(faces) != 1:
            errors.append(
                f"{image_path}: Detected {len(faces)} faces"
            )
            continue

        embedding = faces[0].embedding

        if embedding.shape != (512,):
            errors.append(
                f"{image_path}: Invalid embedding shape "
                f"{embedding.shape}"
            )
            continue

        embeddings.append(embedding)
        participants.append(participant)
        image_names.append(
            f"{participant}/{image_path.name}"
        )

        print(f"  [OK] {image_path.name} - 512 dimensions")


# =====================================
# 4. VALIDATE AND SAVE
# =====================================

if errors:
    print("\n[ERROR] Some images need to be checked:")

    for error in errors:
        print(" -", error)

    raise SystemExit("Embedding extraction was not completed.")

if not embeddings:
    raise SystemExit("No valid embeddings were extracted.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_FILE,
    embeddings=np.stack(embeddings),
    participants=np.array(participants),
    image_names=np.array(image_names)
)

print("\n" + "=" * 45)
print("EMBEDDING EXTRACTION COMPLETED")
print("=" * 45)

print("Total images:", len(embeddings))
print("Embedding shape:", np.stack(embeddings).shape)
print("Participants:", sorted(set(participants)))
print("Saved to:", OUTPUT_FILE)
