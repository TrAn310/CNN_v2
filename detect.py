import os
import time
import cv2
import threading
import numpy as np

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from ultralytics.engine.results import Boxes


# =====================================================
# 1. CAU HINH
# =====================================================

MA_XAC_THUC = "@NCKHSV2026"

MODEL_PATH = "best.onnx"
IP = "192.168.100.120"

URL = (
    f"rtsp://admin:{MA_XAC_THUC}"
    f"@{IP}:554/h264/ch1/sub/av_stream"
)

IMGSZ = 416
CONF = 0.7

PERSON_CLASS_ID = 5

WARMUP_FRAMES = 5
REPORT_INTERVAL = 100

WINDOW_NAME = "PPE Detection - Person Tracking"

# Mau OpenCV theo thu tu BGR
COLOR_PPE_OK = (0, 255, 0)       # Xanh la: co trang bi bao ho
COLOR_PPE_VIOLATION = (0, 0, 255)  # Do: khong co trang bi bao ho
COLOR_PERSON = (255, 180, 0)     # Xanh duong: Person + tracking ID
COLOR_OTHER = (0, 220, 255)      # Vang: cac doi tuong khac

# Cac class PPE theo ID cua model
PPE_OK_CLASSES = {0, 1, 7}       # Hardhat, Mask, Safety Vest
PPE_VIOLATION_CLASSES = {2, 3, 4}  # NO-Hardhat, NO-Mask, NO-Safety Vest

# Giao dien FPS nho gon
FPS_FONT_SCALE = 0.45
FPS_THICKNESS = 1
FPS_LINE_GAP = 22


# =====================================================
# 2. CAU HINH OPENCV
# =====================================================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


# =====================================================
# 3. DOC CAMERA RTSP BANG THREAD
# =====================================================

class RTSPReader:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise SystemExit(
                "Khong mo duoc camera RTSP. "
                "Kiem tra IP, URL va ma xac thuc."
            )

        self.frame = None
        self.ok = False
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True
        )
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()

            if not ok:
                time.sleep(0.01)
                continue

            with self.lock:
                self.ok = True
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None

            return self.ok, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# =====================================================
# 4. HAM VE NHAN VA BOUNDING BOX
# =====================================================

def draw_detection(image, box, label, color):
    """Ve bounding box va nhan voi mau duoc chi dinh."""
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # Dat nhan phia tren bounding box neu du cho
    text_y = y1 - 8
    if text_y < 18:
        text_y = y1 + 20

    cv2.putText(
        image,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA
    )


def draw_fps_panel(image, inference_fps, processing_fps):
    """Hien thi hai thong so FPS nho gon o goc tren ben phai."""
    panel_width = 205
    panel_height = 52

    image_height, image_width = image.shape[:2]

    # Can le phai, cach mep tren va mep phai 5 pixel
    x_left = max(5, image_width - panel_width - 5)
    y_top = 5

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x_left, y_top),
        (x_left + panel_width, y_top + panel_height),
        (0, 0, 0),
        -1
    )

    # Nen den trong suot de chu de doc hon
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

    cv2.putText(
        image,
        f"Inference FPS: {inference_fps:.2f}",
        (x_left + 7, y_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        FPS_FONT_SCALE,
        (0, 255, 0),
        FPS_THICKNESS,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        f"Processing FPS: {processing_fps:.2f}",
        (x_left + 7, y_top + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        FPS_FONT_SCALE,
        (255, 220, 0),
        FPS_THICKNESS,
        cv2.LINE_AA
    )


# =====================================================
# 5. KHOI TAO YOLO
# =====================================================

print("\nDang khoi tao mo hinh YOLO...")

model = YOLO(
    MODEL_PATH,
    task="detect"
)

print("Da tai mo hinh thanh cong!")
print("\nCac lop cua model:")

for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")


# =====================================================
# 6. KHOI TAO BYTETRACK
# =====================================================

print("\nDang khoi tao ByteTrack...")

tracker_config = YAML.load(
    check_yaml("bytetrack.yaml")
)

tracker_args = IterableSimpleNamespace(
    **tracker_config
)

tracker = BYTETracker(tracker_args)

print("Da khoi tao ByteTrack!")
print("Chi tracking Person (class ID = 5)")


# =====================================================
# 7. KET NOI CAMERA
# =====================================================

print("\nDang ket noi camera RTSP...")

reader = RTSPReader(URL)

print("Da ket noi camera thanh cong!")


# =====================================================
# 8. LAY FRAME DAU TIEN
# =====================================================

print("\nDang cho frame dau tien...")

frame = None

while frame is None:
    ok, frame = reader.read()

    if not ok or frame is None:
        frame = None
        time.sleep(0.01)


# =====================================================
# 9. WARM-UP MO HINH
# =====================================================

print("Dang warm-up mo hinh...")

for i in range(WARMUP_FRAMES):
    model.predict(
        frame,
        imgsz=IMGSZ,
        conf=CONF,
        verbose=False
    )

    print(f"Warm-up: {i + 1}/{WARMUP_FRAMES}")

print("Warm-up hoan tat!\n")


# =====================================================
# 10. KHOI TAO BIEN THONG KE
# =====================================================

count = 0

total_inference = 0.0
total_processing = 0.0

min_inference = float("inf")
max_inference = 0.0

min_processing = float("inf")
max_processing = 0.0


# =====================================================
# 11. CHUONG TRINH CHINH
# =====================================================

print("BAT DAU NHAN DIEN PPE + PERSON TRACKING")
print("Nhan Q de thoat chuong trinh.\n")

try:
    while True:
        loop_start = time.perf_counter()

        # Doc frame moi nhat
        ok, frame = reader.read()

        if not ok or frame is None:
            time.sleep(0.01)
            continue

        # Chay YOLO mot lan tren moi frame
        t0 = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False
        )

        inference_time = time.perf_counter() - t0
        result = results[0]

        # Tach Person khoi cac object khac de dua vao ByteTrack
        if result.boxes is not None:
            person_mask = result.boxes.cls == PERSON_CLASS_ID
            person_detections = result.boxes[person_mask]
        else:
            person_detections = Boxes(
                np.empty((0, 6), dtype=np.float32),
                frame.shape[:2]
            )

        # Chi tracking Person
        tracked_persons = tracker.update(
            person_detections,
            frame
        )

        annotated = frame.copy()

        # Ve cac object khong phai Person
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())

                # Person se duoc ve rieng bang ByteTrack
                if class_id == PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_name = model.names[class_id]

                label = f"{class_name} {confidence:.2f}"

                # Mau theo trang thai PPE
                if class_id in PPE_OK_CLASSES:
                    box_color = COLOR_PPE_OK
                elif class_id in PPE_VIOLATION_CLASSES:
                    box_color = COLOR_PPE_VIOLATION
                else:
                    box_color = COLOR_OTHER

                draw_detection(
                    annotated,
                    (x1, y1, x2, y2),
                    label,
                    box_color
                )

        # Ve Person kem tracking ID
        for track in tracked_persons:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            confidence = float(track[5])

            label = f"Person ID: {track_id} {confidence:.2f}"

            draw_detection(
                annotated,
                (x1, y1, x2, y2),
                label,
                COLOR_PERSON
            )

        # Tinh thoi gian xu ly
        processing_time = time.perf_counter() - loop_start

        inference_fps = 1.0 / max(inference_time, 1e-6)
        processing_fps = 1.0 / max(processing_time, 1e-6)

        # Chi hien thi hai thong so FPS
        draw_fps_panel(
            annotated,
            inference_fps,
            processing_fps
        )

        cv2.imshow(
            WINDOW_NAME,
            annotated
        )

        # Cap nhat thong ke
        count += 1

        total_inference += inference_time
        total_processing += processing_time

        min_inference = min(min_inference, inference_time)
        max_inference = max(max_inference, inference_time)

        min_processing = min(min_processing, processing_time)
        max_processing = max(max_processing, processing_time)

        # Bao cao sau moi 100 frame
        if count >= REPORT_INTERVAL:
            avg_inference = total_inference / count
            avg_processing = total_processing / count

            print("\n================================")
            print("KET QUA THONG KE 100 FRAME")
            print("================================")

            print(
                f"Inference trung binh: "
                f"{avg_inference:.4f} giay"
            )

            print(
                f"Processing trung binh: "
                f"{avg_processing:.4f} giay"
            )

            print(
                f"Inference FPS: "
                f"{1.0 / avg_inference:.2f}"
            )

            print(
                f"Processing FPS: "
                f"{1.0 / avg_processing:.2f}"
            )

            print(
                f"Inference nhanh nhat: "
                f"{min_inference:.4f} giay"
            )

            print(
                f"Inference cham nhat: "
                f"{max_inference:.4f} giay"
            )

            print("================================\n")

            # Reset thong ke
            count = 0

            total_inference = 0.0
            total_processing = 0.0

            min_inference = float("inf")
            max_inference = 0.0

            min_processing = float("inf")
            max_processing = 0.0

        # Kiem tra phim Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nDang dung chuong trinh...")
            break

except KeyboardInterrupt:
    print("\nNguoi dung dung chuong trinh.")

except Exception as e:
    print("\nLOI CHUONG TRINH:")
    print(e)

    import traceback
    traceback.print_exc()

finally:
    reader.release()
    cv2.destroyAllWindows()
    print("Da dong camera va giai phong tai nguyen.")
