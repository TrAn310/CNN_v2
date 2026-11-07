import os

import csv

import time

import threading

from collections import deque, Counter

from datetime import datetime

from pathlib import Path



import cv2
import numpy as np
import yaml

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from ultralytics.utils.checks import check_yaml

from ultralytics.engine.results import Boxes





# =====================================================

# 1. CAU HINH

# =====================================================



MA_XAC_THUC = "@NCKHSV2026"



MODEL_PATH = "best.onnx"

IP = "192.168.100.120"



URL = f"rtsp://admin:{MA_XAC_THUC}@{IP}:554/h264/ch1/sub/av_stream"



IMGSZ = 416

CONF = 0.7



PERSON_CLASS_ID = 5



WARMUP_FRAMES = 5

REPORT_INTERVAL = 100



WINDOW_NAME = "PPE Detection - Person Tracking"



# Mau OpenCV theo thu tu BGR

COLOR_PPE_OK = (0, 255, 0)           # Xanh la: co trang bi bao ho

COLOR_PPE_VIOLATION = (0, 0, 255)    # Do: khong co trang bi bao ho

COLOR_PERSON = (255, 180, 0)         # Xanh duong: Person chua ket luan

COLOR_PERSON_OK = (0, 200, 0)        # Xanh la: Person dat day du PPE

COLOR_PERSON_ALERT = (0, 0, 255)     # Do: Person dang bi canh bao

COLOR_OTHER = (0, 220, 255)          # Vang: cac doi tuong khac


# Bang mau rieng cho Person ID (BGR).
# Cac mau nay khong dung xanh la/do de tranh nham voi class PPE.
PERSON_COLOR_PALETTE = [
    (255, 0, 255),    # Hong tim
    (255, 128, 0),    # Xanh duong dam
    (128, 0, 255),    # Tim
    (0, 165, 255),    # Cam
    (255, 255, 0),    # Xanh cyan
    (180, 105, 255),  # Hong
    (128, 128, 0),    # Xanh teal
    (255, 0, 128),    # Tim hong
    (0, 200, 255),    # Vang cam
    (200, 80, 160),   # Tim hong dam
    (255, 180, 80),   # Xanh nhat
    (80, 120, 255),   # Cam hong
]


def get_person_color(track_id):
    """Tra ve mau co dinh theo Person ID, khong phu thuoc trang thai PPE."""
    return PERSON_COLOR_PALETTE[(int(track_id) - 1) % len(PERSON_COLOR_PALETTE)]



# Cac class PPE theo ID cua model

PPE_OK_CLASSES = {0, 1, 7}           # Hardhat, Mask, Safety Vest

PPE_VIOLATION_CLASSES = {2, 3, 4}    # NO-Hardhat, NO-Mask, NO-Safety Vest



# Giao dien FPS nho gon

FPS_FONT_SCALE = 0.45

FPS_THICKNESS = 1



# ---------- Logic canh bao (buoc 5) ----------

GROUPS = ("Mu", "Khau trang", "Ao bao ho")



SMOOTH_FRAMES = 5          # so frame dung de bo phieu da so cho moi trang thai

VIOLATION_SECONDS = 2.0    # vi pham lien tuc bao lau thi moi canh bao

ALERT_COOLDOWN = 30.0      # sau khi canh bao 1 nguoi, doi bao lau moi bao lai

TRACK_TIMEOUT = 10.0       # quen Person ID neu khong thay trong ngan nay giay



# Nhom nao bi coi la vi pham de canh bao. Bo "Khau trang" neu khong can.

ALERT_GROUPS = ("Mu", "Khau trang", "Ao bao ho")



ALERT_DIR = "alerts"

LOG_FILE = os.path.join(ALERT_DIR, "alert_log.csv")





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



        self.thread = threading.Thread(target=self._loop, daemon=True)

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

# 4. HAM VE NHAN, BOUNDING BOX VA DANH GIA PPE

# =====================================================



def draw_detection(image, box, label, color):

    """Ve bounding box va nhan voi mau duoc chi dinh."""

    x1, y1, x2, y2 = map(int, box)



    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)



    text_y = y1 - 8

    if text_y < 18:

        text_y = y1 + 20



    cv2.putText(

        image, label, (x1, text_y),

        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA

    )





def draw_fps_panel(image, inference_fps, processing_fps):

    """Hien thi hai thong so FPS nho gon o goc tren ben phai."""

    panel_width = 205

    panel_height = 52



    image_height, image_width = image.shape[:2]



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

    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)



    cv2.putText(

        image, f"Inference FPS: {inference_fps:.2f}",

        (x_left + 7, y_top + 20),

        cv2.FONT_HERSHEY_SIMPLEX, FPS_FONT_SCALE,

        (0, 255, 0), FPS_THICKNESS, cv2.LINE_AA

    )



    cv2.putText(

        image, f"Processing FPS: {processing_fps:.2f}",

        (x_left + 7, y_top + 42),

        cv2.FONT_HERSHEY_SIMPLEX, FPS_FONT_SCALE,

        (255, 220, 0), FPS_THICKNESS, cv2.LINE_AA

    )





def get_ppe_group(class_id):

    """

    Tra ve nhom PPE va trang thai:

    True = co trang bi, False = khong co trang bi.

    """

    ppe_mapping = {

        0: ("Mu", True),              # Hardhat

        2: ("Mu", False),             # NO-Hardhat

        1: ("Khau trang", True),      # Mask

        3: ("Khau trang", False),     # NO-Mask

        7: ("Ao bao ho", True),       # Safety Vest

        4: ("Ao bao ho", False),      # NO-Safety Vest

    }



    return ppe_mapping.get(class_id)





def associate_ppe_to_person(person_box, ppe_box):

    """Gan PPE cho Person neu tam khung PPE nam trong khung Person."""

    px1, py1, px2, py2 = person_box

    x1, y1, x2, y2 = ppe_box



    center_x = (x1 + x2) / 2.0

    center_y = (y1 + y2) / 2.0



    return px1 <= center_x <= px2 and py1 <= center_y <= py2





def evaluate_person_ppe(person_box, ppe_detections):

    """

    Danh gia PPE cua mot Person trong MOT frame.

    Neu co nhieu detection trong cung nhom, chon detection co confidence cao nhat.

    Neu khong co detection thi de trang thai None (chua xac dinh).

    """

    candidates = {name: [] for name in GROUPS}



    for detection in ppe_detections:

        group_info = get_ppe_group(detection["class_id"])



        if group_info is None:

            continue



        group_name, is_equipped = group_info



        if associate_ppe_to_person(person_box, detection["box"]):

            candidates[group_name].append({

                "is_equipped": is_equipped,

                "confidence": detection["confidence"],

            })



    statuses = {}



    for group_name, group_candidates in candidates.items():

        if not group_candidates:

            statuses[group_name] = None

        else:

            best = max(group_candidates, key=lambda item: item["confidence"])

            statuses[group_name] = best["is_equipped"]



    return statuses





def draw_person_id(image, person_box, track_id, confidence, color):

    """Ve bounding box va Person ID tren cua so camera."""

    x1, y1, x2, y2 = map(int, person_box)



    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)



    person_label = f"Person ID: {track_id} {confidence:.2f}"

    label_y = y1 - 8 if y1 >= 22 else y1 + 20



    cv2.putText(

        image, person_label, (x1, label_y),

        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA

    )





def print_person_status(track_id, statuses):

    """In trang thai PPE (da lam muot) cua mot Person ra Terminal."""

    status_map = {True: "DAT", False: "VI PHAM", None: "CHUA XAC DINH"}



    print("\n" + "=" * 48)

    print(f"PERSON ID: {track_id}")

    print("-" * 48)



    for group_name in GROUPS:

        print(f"{group_name:<20}: {status_map[statuses[group_name]]}")



    if any(status is False for status in statuses.values()):

        overall = "KHONG DAT"

    elif any(status is None for status in statuses.values()):

        overall = "CHUA DAY DU DU LIEU"

    else:

        overall = "DAT DAY DU PPE"



    print("-" * 48)

    print(f"TRANG THAI CHUNG: {overall}")

    print("=" * 48, flush=True)





# =====================================================

# 5. LOGIC CANH BAO (BUOC 5)

# =====================================================



def majority(values, previous):

    """

    Bo phieu da so tren cac gia tri True/False, bo qua None.

    Khong co gia tri nao -> None. Hoa phieu -> giu gia tri truoc do.

    """

    observed = [v for v in values if v is not None]



    if not observed:

        return None



    counts = Counter(observed)



    if counts[True] == counts[False]:

        return previous



    return counts[True] > counts[False]





class PersonState:

    """Luu trang thai PPE da lam muot va dong ho vi pham cua mot Person ID."""



    def __init__(self, now):

        self.history = {g: deque(maxlen=SMOOTH_FRAMES) for g in GROUPS}

        self.smoothed = {g: None for g in GROUPS}

        self.violation_since = {g: None for g in GROUPS}

        self.last_seen = now

        self.last_alert = None

        self.last_printed = None

        self.alerting = False



    def update(self, statuses, now):

        self.last_seen = now



        for g in GROUPS:

            self.history[g].append(statuses[g])

            self.smoothed[g] = majority(self.history[g], self.smoothed[g])



            if self.smoothed[g] is False:

                if self.violation_since[g] is None:

                    self.violation_since[g] = now

            elif self.smoothed[g] is True:

                self.violation_since[g] = None

            # None (khong thay) thi giu nguyen dong ho



    def snapshot(self):

        return tuple(self.smoothed[g] for g in GROUPS)



    def confirmed_violations(self, now):

        """Cac nhom da vi pham lien tuc >= VIOLATION_SECONDS."""

        return [

            g for g in ALERT_GROUPS

            if self.violation_since[g] is not None

            and now - self.violation_since[g] >= VIOLATION_SECONDS

        ]



    def any_violation_running(self):

        return any(self.violation_since[g] is not None for g in ALERT_GROUPS)



    def color(self):

        if self.alerting:

            return COLOR_PERSON_ALERT

        if all(self.smoothed[g] is True for g in GROUPS):

            return COLOR_PERSON_OK

        return COLOR_PERSON





class AlertManager:

    def __init__(self):

        self.states = {}

        os.makedirs(ALERT_DIR, exist_ok=True)



        if not os.path.exists(LOG_FILE):

            with open(LOG_FILE, "w", newline="", encoding="utf-8-sig") as f:

                csv.writer(f).writerow(

                    ["thoi_gian", "person_id", "vi_pham", "anh"]

                )



    def update(self, track_id, statuses, now):

        state = self.states.get(track_id)



        if state is None:

            state = PersonState(now)

            self.states[track_id] = state



        state.update(statuses, now)

        return state



    def check_alert(self, track_id, state, now):

        """

        Tra ve danh sach nhom vi pham neu can canh bao ngay bay gio,

        nguoc lai tra ve danh sach rong.

        """

        violations = state.confirmed_violations(now)



        if violations:

            in_cooldown = (

                state.last_alert is not None

                and now - state.last_alert < ALERT_COOLDOWN

            )

            state.alerting = True



            if not in_cooldown:

                return violations



            return []



        # Khong con vi pham nao dang chay -> da khac phuc

        if state.alerting and not state.any_violation_running():

            state.alerting = False

            state.last_alert = None

            print(

                f"\n[{datetime.now():%H:%M:%S}] "

                f"Person ID {track_id} da khac phuc, dat PPE.",

                flush=True

            )



        return []



    def fire(self, image, track_id, violations, now):

        state = self.states[track_id]

        state.last_alert = now



        stamp = datetime.now()

        stamp = datetime.now()

        filename = f"{stamp:%Y%m%d_%H%M%S}_ID{track_id}.jpg"

        image_path = os.path.join(ALERT_DIR, filename)



        cv2.imwrite(image_path, image)



        violation_text = ", ".join(violations)



        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:

            csv.writer(f).writerow([

                f"{stamp:%Y-%m-%d %H:%M:%S}",

                track_id,

                violation_text,

                image_path

            ])



        print("\n" + "!" * 48)

        print(f"CANH BAO [{stamp:%H:%M:%S}]  Person ID {track_id}")

        print(f"Thieu: {violation_text}")

        print(f"Anh bang chung: {image_path}")

        print("!" * 48, flush=True)



    def cleanup(self, now):

        stale = [

            track_id for track_id, state in self.states.items()

            if now - state.last_seen > TRACK_TIMEOUT

        ]



        for track_id in stale:

            del self.states[track_id]





# =====================================================

# 6. KHOI TAO YOLO

# =====================================================



print("\nDang khoi tao mo hinh YOLO...")



model = YOLO(MODEL_PATH, task="detect")



print("Da tai mo hinh thanh cong!")

print("\nCac lop cua model:")



for class_id, class_name in model.names.items():

    print(f"{class_id}: {class_name}")





# =====================================================

# 7. KHOI TAO BYTETRACK VA ALERT MANAGER

# =====================================================



print("\nDang khoi tao ByteTrack...")



tracker_config = yaml.safe_load(
    Path(check_yaml("bytetrack.yaml")).read_text(encoding="utf-8")
)

tracker_args = IterableSimpleNamespace(**tracker_config)

tracker = BYTETracker(tracker_args)



print("Da khoi tao ByteTrack!")

print("Chi tracking Person (class ID = 5)")



alerts = AlertManager()



print(f"Canh bao: vi pham lien tuc {VIOLATION_SECONDS}s, "

      f"cooldown {ALERT_COOLDOWN}s")

print(f"Anh va log luu tai thu muc: {ALERT_DIR}/")





# =====================================================

# 8. KET NOI CAMERA

# =====================================================



print("\nDang ket noi camera RTSP...")



reader = RTSPReader(URL)



print("Da ket noi camera thanh cong!")





# =====================================================

# 9. LAY FRAME DAU TIEN

# =====================================================



print("\nDang cho frame dau tien...")



frame = None



while frame is None:

    ok, frame = reader.read()



    if not ok or frame is None:

        frame = None

        time.sleep(0.01)



print(f"Do phan giai frame: {frame.shape[1]} x {frame.shape[0]}")





# =====================================================

# 10. WARM-UP MO HINH

# =====================================================



print("Dang warm-up mo hinh...")



for i in range(WARMUP_FRAMES):

    model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)

    print(f"Warm-up: {i + 1}/{WARMUP_FRAMES}")



print("Warm-up hoan tat!\n")





# =====================================================

# 11. KHOI TAO BIEN THONG KE

# =====================================================



count = 0



total_inference = 0.0

total_processing = 0.0



min_inference = float("inf")

max_inference = 0.0



min_processing = float("inf")

max_processing = 0.0





# =====================================================

# 12. CHUONG TRINH CHINH

# =====================================================



print("BAT DAU NHAN DIEN PPE + PERSON TRACKING + CANH BAO")

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

            frame, imgsz=IMGSZ, conf=CONF, verbose=False

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

        tracked_persons = tracker.update(person_detections, frame)



        annotated = frame.copy()



        # Thu thap tat ca detection PPE de lien ket voi tung Person

        ppe_detections = []



        # Ve cac object khong phai Person

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0].item())



                # Person se duoc ve rieng bang ByteTrack

                if class_id == PERSON_CLASS_ID:

                    continue



                box_coords = box.xyxy[0].tolist()

                confidence = float(box.conf[0].item())

                class_name = model.names[class_id]



                label = f"{class_name} {confidence:.2f}"



                if class_id in PPE_OK_CLASSES:

                    box_color = COLOR_PPE_OK

                elif class_id in PPE_VIOLATION_CLASSES:

                    box_color = COLOR_PPE_VIOLATION

                else:

                    box_color = COLOR_OTHER



                draw_detection(annotated, box_coords, label, box_color)



                if get_ppe_group(class_id) is not None:

                    ppe_detections.append({

                        "class_id": class_id,

                        "class_name": class_name,

                        "box": box_coords,

                        "confidence": confidence

                    })



        # Danh gia PPE + logic canh bao cho tung Person ID

        now = time.perf_counter()

        pending_alerts = []



        for track in tracked_persons:

            person_box = list(map(float, track[:4]))

            track_id = int(track[4])

            confidence = float(track[5])



            statuses = evaluate_person_ppe(person_box, ppe_detections)



            # Lam muot qua nhieu frame, cap nhat dong ho vi pham

            state = alerts.update(track_id, statuses, now)



            # In Terminal khi trang thai DA LAM MUOT thay doi

            if state.last_printed != state.snapshot():

                print_person_status(track_id, state.smoothed)

                state.last_printed = state.snapshot()



            # Kiem tra co can canh bao khong

            violations = alerts.check_alert(track_id, state, now)



            if violations:

                pending_alerts.append((track_id, violations))



            draw_person_id(

                annotated, person_box, track_id, confidence,
                get_person_color(track_id)

            )



        # Luu anh + log sau khi da ve xong tat ca khung

        for track_id, violations in pending_alerts:

            alerts.fire(annotated, track_id, violations, now)



        alerts.cleanup(now)



        # Tinh thoi gian xu ly

        processing_time = time.perf_counter() - loop_start



        inference_fps = 1.0 / max(inference_time, 1e-6)

        processing_fps = 1.0 / max(processing_time, 1e-6)



        draw_fps_panel(annotated, inference_fps, processing_fps)



        cv2.imshow(WINDOW_NAME, annotated)



        # Cap nhat thong ke

        count += 1



        total_inference += inference_time

        total_processing += processing_time



        min_inference = min(min_inference, inference_time)

        max_inference = max(max_inference, inference_time)



        min_processing = min(min_processing, processing_time)

        max_processing = max(max_processing, processing_time)



        # Bao cao sau moi REPORT_INTERVAL frame

        if count >= REPORT_INTERVAL:

            avg_inference = total_inference / count

            avg_processing = total_processing / count



            print("\n================================")

            print(f"KET QUA THONG KE {REPORT_INTERVAL} FRAME")

            print("================================")

            print(f"Inference trung binh: {avg_inference:.4f} giay")

            print(f"Processing trung binh: {avg_processing:.4f} giay")

            print(f"Inference FPS: {1.0 / avg_inference:.2f}")

            print(f"Processing FPS: {1.0 / avg_processing:.2f}")

            print(f"Inference nhanh nhat: {min_inference:.4f} giay")

            print(f"Inference cham nhat: {max_inference:.4f} giay")

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