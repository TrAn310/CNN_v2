
import cv2
import numpy as np
import argparse
from pathlib import Path
from insightface.app import FaceAnalysis


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "buffalo_l"
THRESHOLD = 0.32
DETECTION_SIZE = (640, 640)


# ==========================================
# INITIALIZE MODEL
# ==========================================

def initialize_model():
    print("[INFO] Initializing InsightFace...")

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    print("[INFO] Model initialized successfully.")

    return app


# ==========================================
# FACE DETECTION AND FEATURE EXTRACTION
# ==========================================

def extract_face_embedding(app, image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    faces = app.get(image)

    print(f"\n[INFO] Image: {image_path.name}")
    print(f"[INFO] Detected faces: {len(faces)}")

    if len(faces) == 0:
        raise ValueError(
            f"No face detected in {image_path.name}"
        )

    if len(faces) > 1:
        raise ValueError(
            f"Multiple faces detected in {image_path.name}. "
            "Please use an image containing only one face."
        )

    face = faces[0]

    embedding = np.asarray(
        face.embedding,
        dtype=np.float32
    )

    bbox = face.bbox
    detection_score = float(face.det_score)

    print(f"[INFO] Detection score: {detection_score:.4f}")
    print(f"[INFO] Bounding box: {bbox}")
    print(f"[INFO] Embedding shape: {embedding.shape}")

    return embedding


# ==========================================
# COSINE SIMILARITY
# ==========================================

def calculate_cosine_similarity(embedding1, embedding2):

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Invalid zero embedding.")

    similarity = np.dot(
        embedding1,
        embedding2
    ) / (norm1 * norm2)

    return float(similarity)


# ==========================================
# FACE VERIFICATION
# ==========================================

def verify_faces(app, image1_path, image2_path):

    print("\n" + "=" * 55)
    print("FACE VERIFICATION")
    print("=" * 55)

    embedding1 = extract_face_embedding(
        app,
        image1_path
    )

    embedding2 = extract_face_embedding(
        app,
        image2_path
    )

    similarity = calculate_cosine_similarity(
        embedding1,
        embedding2
    )

    print("\n" + "=" * 55)
    print("VERIFICATION RESULT")
    print("=" * 55)

    print(f"Cosine Similarity: {similarity:.6f}")
    print(f"Threshold:         {THRESHOLD:.2f}")

    if similarity >= THRESHOLD:
        print("Result: SAME PERSON (similarity criterion met)")
    else:
        print("Result: DIFFERENT PERSON (similarity criterion not met)")

    print("=" * 55)

    return similarity


# ==========================================
# MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Offline two-image face verification"
    )

    parser.add_argument(
        "--image1",
        required=True,
        help="Path to first image"
    )

    parser.add_argument(
        "--image2",
        required=True,
        help="Path to second image"
    )

    args = parser.parse_args()

    try:
        app = initialize_model()

        verify_faces(
            app,
            args.image1,
            args.image2
        )

    except Exception as error:
        print(f"\n[ERROR] {error}")


if __name__ == "__main__":
    main()
