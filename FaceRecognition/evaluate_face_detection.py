"""
Batch evaluation for non-identifying InsightFace face detection.

Folder layout:
FaceRecognition/
  evaluate_face_detection.py
  dataset/
    images/       # Put test images here
    labels.csv    # Optional manual face counts
  evaluation_results/  # Created automatically

labels.csv columns:
image,actual_faces,condition
test1.jpg,1,frontal
test2.jpg,7,group

The script compares face counts per image only. This is NOT per-face
Precision/Recall because it does not match each detection to ground-truth boxes.
"""

from pathlib import Path
import csv
import cv2
from insightface.app import FaceAnalysis

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "dataset" / "images"
LABEL_FILE = BASE_DIR / "dataset" / "labels.csv"
OUTPUT_DIR = BASE_DIR / "evaluation_results"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_labels():
    labels = {}
    if not LABEL_FILE.exists():
        print(f"[INFO] No labels file found: {LABEL_FILE}")
        print("[INFO] Detection will run, but count comparison will be skipped.")
        return labels

    with LABEL_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"image", "actual_faces", "condition"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "labels.csv must contain columns: image, actual_faces, condition"
            )

        for row in reader:
            filename = row["image"].strip()
            if not filename:
                continue
            actual_text = row["actual_faces"].strip()
            labels[filename] = {
                "actual_faces": int(actual_text) if actual_text else None,
                "condition": row["condition"].strip(),
            }
    return labels


def draw_detections(image, faces):
    result = image.copy()
    for index, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = map(int, face.bbox)
        score = float(face.det_score)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            result, f"Face {index}: {score:.2f}",
            (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2, cv2.LINE_AA
        )

    cv2.rectangle(result, (8, 8), (270, 42), (0, 0, 0), -1)
    cv2.putText(
        result, f"Detected faces: {len(faces)}", (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    return result


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Missing folder: {IMAGE_DIR}\n"
            "Create dataset/images and copy your test images into it."
        )

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    print("[INFO] Initializing buffalo_l on CPU...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[INFO] Model ready.\n")

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")
        image = cv2.imread(str(image_path))

        label = labels.get(image_path.name, {})
        actual = label.get("actual_faces")
        condition = label.get("condition", "")

        if image is None:
            print("  ERROR: Could not read image.")
            rows.append({
                "image": image_path.name,
                "condition": condition,
                "actual_faces": "" if actual is None else actual,
                "detected_faces": "",
                "count_difference": "",
                "count_match": "IMAGE_READ_ERROR",
                "status": "IMAGE_READ_ERROR",
            })
            continue

        faces = app.get(image)
        for i, face in enumerate(faces):
            print(f"\nKhuôn mặt {i + 1}")
            print("Bounding box:", face.bbox)
            print("Detection score:", face.det_score)
            print("Embedding shape:", face.embedding.shape)
            print("Embedding:")
            print(face.embedding)
        detected = len(faces)

        output_path = ANNOTATED_DIR / f"{image_path.stem}_detected{image_path.suffix}"
        cv2.imwrite(str(output_path), draw_detections(image, faces))

        if actual is None:
            difference = ""
            count_match = "NOT_LABELED"
            status = "DETECTION_ONLY"
        else:
            difference = abs(detected - actual)
            count_match = "YES" if detected == actual else "NO"
            status = "COUNT_MATCH" if count_match == "YES" else "COUNT_MISMATCH"

        rows.append({
            "image": image_path.name,
            "condition": condition,
            "actual_faces": "" if actual is None else actual,
            "detected_faces": detected,
            "count_difference": difference,
            "count_match": count_match,
            "status": status,
        })

        print(f"  Detected faces: {detected}")
        if actual is not None:
            print(f"  Actual faces:   {actual}")
            print(f"  Count matches:  {count_match}")
        print(f"  Saved: {output_path}\n")

    report_path = OUTPUT_DIR / "face_detection_report.csv"
    columns = [
        "image", "condition", "actual_faces", "detected_faces",
        "count_difference", "count_match", "status"
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    labeled = [
        row for row in rows
        if row["actual_faces"] != "" and row["status"] != "IMAGE_READ_ERROR"
    ]
    matched = sum(row["count_match"] == "YES" for row in labeled)

    print("=" * 55)
    print("FACE DETECTION EVALUATION")
    print("=" * 55)
    print(f"Images found:              {len(image_paths)}")
    print(f"Images processed:          {len(rows)}")
    print(f"Images with manual labels: {len(labeled)}")
    print(f"Images with matching count:{matched:>5}")
    if labeled:
        print(f"Count-match rate:          {matched / len(labeled) * 100:.2f}%")
        print("This is image-level count agreement, NOT Precision or Recall.")
    print(f"\nCSV report: {report_path}")
    print(f"Annotated images: {ANNOTATED_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
