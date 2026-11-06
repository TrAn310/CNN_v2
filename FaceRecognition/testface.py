import cv2
from insightface.app import FaceAnalysis

# Khởi tạo mô hình
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

# Đọc ảnh thử nghiệm
image1 = cv2.imread("test8.jpg")
image = cv2.resize(image1,(800,800))


if image is None:
    raise FileNotFoundError("Không tìm thấy ảnh test.jpg")

# Phát hiện khuôn mặt
faces = app.get(image)

print("Số khuôn mặt phát hiện:", len(faces))

# Vẽ khung quanh khuôn mặt
for face in faces:
    x1, y1, x2, y2 = map(int, face.bbox)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

