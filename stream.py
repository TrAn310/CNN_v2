import os, cv2, threading

# Dùng TCP để ổn định hơn UDP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

URL = "rtsp://admin:@NCKHSV2026@192.168.100.120:554/h264/ch1/sub/av_stream"

class RTSPReader:
    """Luôn giữ frame mới nhất để hình không bị trễ tích luỹ."""
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise SystemExit("Không mở được stream: kiểm tra URL / mã xác thực")
        self.frame, self.ok = None, False
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            ok, f = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, f

    def read(self):
        with self.lock:
            return self.ok, (None if self.frame is None else self.frame.copy())

reader = RTSPReader(URL)
while True:
    ok, frame = reader.read()
    if ok and frame is not None:
        cv2.imshow("camera", frame)   # sau này thay bằng YOLO ở đây
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()