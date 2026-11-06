
import ast
import os
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

EMBEDDINGS_FILE = (
    BASE_DIR
    / "evaluation_results"
    / "verification_embeddings.npz"
)

DETECT_V4_FILE = PROJECT_DIR / "detect_v4.py"

GALLERY_SIZE = 4
THRESHOLD = 0.32

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)

WINDOW_NAME = "IP CAMERA - FACE RECOGNITION"


# ============================================================
# 2. READ CAMERA CONFIG WITHOUT RUNNING detect_v4.py
# ============================================================

def load_camera_config():

    if not DETECT_V4_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find detect_v4.py: {DETECT_V4_FILE}"
        )

    # Parse source code only. Do not execute detect_v4.py.
    source = DETECT_V4_FILE.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    config = {}

    for node in ast.walk(tree):

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if (
                    isinstance(target, ast.Name)
                    and target.id in ("MA_XAC_THUC", "IP")
                ):

                    try:
                        config[target.id] = ast.literal_eval(
                            node.value
                        )
                    except (ValueError, TypeError):
                        pass

        elif isinstance(node, ast.AnnAssign):

            if (
                isinstance(node.target, ast.Name)
                and node.target.id in ("MA_XAC_THUC", "IP")
                and node.value is not None
            ):

                try:
                    config[node.target.id] = ast.literal_eval(
                        node.value
                    )
                except (ValueError, TypeError):
                    pass

    if "MA_XAC_THUC" not in config or "IP" not in config:
        raise RuntimeError(
            "Cannot find MA_XAC_THUC or IP in detect_v4.py"
        )

    camera_ip = config["IP"]
    camera_password = config["MA_XAC_THUC"]

    if not camera_ip or not camera_password:
        raise RuntimeError(
            "Camera IP or password is empty"
        )

    camera_url = (
        f"rtsp://admin:{camera_password}"
        f"@{camera_ip}:554/h264/ch1/sub/av_stream"
    )

    return camera_url


# ============================================================
# 3. EMBEDDING FUNCTIONS
# ============================================================

def normalize(embedding):

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Zero norm embedding")

    return embedding / norm


def cosine_similarity(a, b):

    return float(
        np.dot(
            normalize(a),
            normalize(b)
        )
    )


# ============================================================
# 4. FACE RECOGNIZER
# ============================================================

class FaceRecognizer:

    def __init__(self):

        print("[INFO] Loading face gallery...")

        self.gallery = self.load_gallery()

        print("[INFO] Initializing InsightFace...")

        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=DET_SIZE
        )

        print("[INFO] Face recognition initialized.")
        print("[INFO] Registered participants:", list(self.gallery))

    def load_gallery(self):

        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"Embedding file not found: {EMBEDDINGS_FILE}"
            )

        data = np.load(EMBEDDINGS_FILE)

        embeddings = data["embeddings"]
        participants = data["participants"]
        image_names = data["image_names"]

        person_images = {}

        for embedding, participant, image_name in zip(
            embeddings,
            participants,
            image_names
        ):

            participant = str(participant)

            person_images.setdefault(
                participant, []
            ).append(
                (str(image_name), embedding)
            )

        gallery = {}

        for participant in sorted(person_images):

            images = sorted(
                person_images[participant],
                key=lambda item: item[0]
            )

            if len(images) < GALLERY_SIZE:
                raise ValueError(
                    f"{participant} has insufficient images"
                )

            reference_embeddings = np.array([
                item[1]
                for item in images[:GALLERY_SIZE]
            ])

            representative = normalize(
                np.mean(
                    reference_embeddings,
                    axis=0
                )
            )

            gallery[participant] = representative

        if not gallery:
            raise ValueError("Gallery is empty")

        return gallery

    def recognize(self, frame):

        if frame is None:
            return []

        faces = self.app.get(frame)

        results = []

        for face in faces:

            embedding = normalize(
                face.embedding.astype(np.float32)
            )

            scores = {
                person: cosine_similarity(
                    embedding,
                    reference
                )
                for person, reference in self.gallery.items()
            }

            best_match = max(
                scores,
                key=scores.get
            )

            best_score = scores[best_match]

            if best_score >= THRESHOLD:
                identity = best_match
            else:
                identity = "Unknown"

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )

            results.append({
                "identity": identity,
                "similarity": best_score,
                "bbox": (x1, y1, x2, y2),
                "det_score": float(face.det_score)
            })

        return results

    @staticmethod
    def draw(frame, results):

        output = frame.copy()

        for result in results:

            x1, y1, x2, y2 = result["bbox"]

            identity = result["identity"]
            similarity = result["similarity"]

            if identity == "Unknown":
                color = (0, 165, 255)
            else:
                color = (0, 200, 0)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = (
                f"{identity} "
                f"({similarity:.2f})"
            )

            cv2.putText(
                output,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return output


# ============================================================
# 5. IP CAMERA
# ============================================================

def open_ip_camera(camera_url):

    os.environ[
        "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    ] = "rtsp_transport;tcp"

    camera = cv2.VideoCapture(
        camera_url,
        cv2.CAP_FFMPEG
    )

    if not camera.isOpened():

        camera.release()

        raise RuntimeError(
            "Cannot connect to EZVIZ IP camera"
        )

    return camera


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IP CAMERA FACE RECOGNITION")
    print("=" * 60)

    # Read existing camera credentials
    camera_url = load_camera_config()

    # Initialize face recognition
    recognizer = FaceRecognizer()

    # Connect IP camera
    print("\n[INFO] Connecting to IP camera...")

    camera = open_ip_camera(camera_url)
    print("Camera width:", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Camera height:", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera FPS:", camera.get(cv2.CAP_PROP_FPS))

    print("[INFO] IP camera connected.")
    print("[INFO] Press Q to exit.")

    frame_count = 0
    read_errors = 0

    recognition_interval = 5
    last_results = []

    start_time = time.time()

    try:

        while True:

            ret, frame = camera.read()

            if not ret:

                read_errors += 1

                if read_errors >= 30:

                    print("[ERROR] Camera stream lost.")
                    break

                continue

            read_errors = 0
            frame_count += 1

            # Chỉ nhận diện mỗi 5 frame
            if frame_count % recognition_interval == 0:
                last_results = recognizer.recognize(frame)

            # Sử dụng kết quả nhận diện gần nhất
            output = recognizer.draw(
                frame,
                last_results
            )

            # FPS
            elapsed = time.time() - start_time

            fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0
            )

            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                WINDOW_NAME,
                output
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("\n[INFO] Camera released.")
        print("[INFO] Face recognition stopped.")


if __name__ == "__main__":

    main()

import ast
import os
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

EMBEDDINGS_FILE = (
    BASE_DIR
    / "evaluation_results"
    / "verification_embeddings.npz"
)

DETECT_V4_FILE = PROJECT_DIR / "detect_v4.py"

GALLERY_SIZE = 4
THRESHOLD = 0.32

MODEL_NAME = "buffalo_l"
DET_SIZE = (320, 320)

WINDOW_NAME = "IP CAMERA - FACE RECOGNITION"


# ============================================================
# 2. READ CAMERA CONFIG WITHOUT RUNNING detect_v4.py
# ============================================================

def load_camera_config():

    if not DETECT_V4_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find detect_v4.py: {DETECT_V4_FILE}"
        )

    # Parse source code only. Do not execute detect_v4.py.
    source = DETECT_V4_FILE.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    config = {}

    for node in ast.walk(tree):

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if (
                    isinstance(target, ast.Name)
                    and target.id in ("MA_XAC_THUC", "IP")
                ):

                    try:
                        config[target.id] = ast.literal_eval(
                            node.value
                        )
                    except (ValueError, TypeError):
                        pass

        elif isinstance(node, ast.AnnAssign):

            if (
                isinstance(node.target, ast.Name)
                and node.target.id in ("MA_XAC_THUC", "IP")
                and node.value is not None
            ):

                try:
                    config[node.target.id] = ast.literal_eval(
                        node.value
                    )
                except (ValueError, TypeError):
                    pass

    if "MA_XAC_THUC" not in config or "IP" not in config:
        raise RuntimeError(
            "Cannot find MA_XAC_THUC or IP in detect_v4.py"
        )

    camera_ip = config["IP"]
    camera_password = config["MA_XAC_THUC"]

    if not camera_ip or not camera_password:
        raise RuntimeError(
            "Camera IP or password is empty"
        )

    camera_url = (
        f"rtsp://admin:{camera_password}"
        f"@{camera_ip}:554/h264/ch1/sub/av_stream"
    )

    return camera_url


# ============================================================
# 3. EMBEDDING FUNCTIONS
# ============================================================

def normalize(embedding):

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Zero norm embedding")

    return embedding / norm


def cosine_similarity(a, b):

    return float(
        np.dot(
            normalize(a),
            normalize(b)
        )
    )


# ============================================================
# 4. FACE RECOGNIZER
# ============================================================

class FaceRecognizer:

    def __init__(self):

        print("[INFO] Loading face gallery...")

        self.gallery = self.load_gallery()

        print("[INFO] Initializing InsightFace...")

        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=DET_SIZE
        )

        print("[INFO] Face recognition initialized.")
        print("[INFO] Registered participants:", list(self.gallery))

    def load_gallery(self):

        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"Embedding file not found: {EMBEDDINGS_FILE}"
            )

        data = np.load(EMBEDDINGS_FILE)

        embeddings = data["embeddings"]
        participants = data["participants"]
        image_names = data["image_names"]

        person_images = {}

        for embedding, participant, image_name in zip(
            embeddings,
            participants,
            image_names
        ):

            participant = str(participant)

            person_images.setdefault(
                participant, []
            ).append(
                (str(image_name), embedding)
            )

        gallery = {}

        for participant in sorted(person_images):

            images = sorted(
                person_images[participant],
                key=lambda item: item[0]
            )

            if len(images) < GALLERY_SIZE:
                raise ValueError(
                    f"{participant} has insufficient images"
                )

            reference_embeddings = np.array([
                item[1]
                for item in images[:GALLERY_SIZE]
            ])

            representative = normalize(
                np.mean(
                    reference_embeddings,
                    axis=0
                )
            )

            gallery[participant] = representative

        if not gallery:
            raise ValueError("Gallery is empty")

        return gallery

    def recognize(self, frame):

        if frame is None:
            return []

        faces = self.app.get(frame)

        results = []

        for face in faces:

            embedding = normalize(
                face.embedding.astype(np.float32)
            )

            scores = {
                person: cosine_similarity(
                    embedding,
                    reference
                )
                for person, reference in self.gallery.items()
            }

            best_match = max(
                scores,
                key=scores.get
            )

            best_score = scores[best_match]

            if best_score >= THRESHOLD:
                identity = best_match
            else:
                identity = "Unknown"

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )

            results.append({
                "identity": identity,
                "similarity": best_score,
                "bbox": (x1, y1, x2, y2),
                "det_score": float(face.det_score)
            })

        return results

    @staticmethod
    def draw(frame, results):

        output = frame.copy()

        for result in results:

            x1, y1, x2, y2 = result["bbox"]

            identity = result["identity"]
            similarity = result["similarity"]

            if identity == "Unknown":
                color = (0, 165, 255)
            else:
                color = (0, 200, 0)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = (
                f"{identity} "
                f"({similarity:.2f})"
            )

            cv2.putText(
                output,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return output


# ============================================================
# 5. IP CAMERA
# ============================================================

def open_ip_camera(camera_url):

    os.environ[
        "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    ] = "rtsp_transport;tcp"

    camera = cv2.VideoCapture(
        camera_url,
        cv2.CAP_FFMPEG
    )

    if not camera.isOpened():

        camera.release()

        raise RuntimeError(
            "Cannot connect to EZVIZ IP camera"
        )

    return camera


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IP CAMERA FACE RECOGNITION")
    print("=" * 60)

    # Read existing camera credentials
    camera_url = load_camera_config()

    # Initialize face recognition
    recognizer = FaceRecognizer()

    # Connect IP camera
    print("\n[INFO] Connecting to IP camera...")

    camera = open_ip_camera(camera_url)

    print("[INFO] IP camera connected.")
    print("[INFO] Press Q to exit.")

    frame_count = 0
    read_errors = 0

    start_time = time.time()

    try:

        while True:

            ret, frame = camera.read()

            if not ret:

                read_errors += 1

                if read_errors >= 30:

                    print("[ERROR] Camera stream lost.")
                    break

                continue

            read_errors = 0
            frame_count += 1

            # Face recognition
            results = recognizer.recognize(frame)

            # Draw results
            output = recognizer.draw(
                frame,
                results
            )

            # FPS
            elapsed = time.time() - start_time

            fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0
            )

            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                WINDOW_NAME,
                output
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("\n[INFO] Camera released.")
        print("[INFO] Face recognition stopped.")


if __name__ == "__main__":

    main()
