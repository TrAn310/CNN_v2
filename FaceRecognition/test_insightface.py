from insightface.app import FaceAnalysis
import onnxruntime as ort

print("ONNX Runtime providers:")
print(ort.get_available_providers())

print("\nInitializing FaceAnalysis...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("\nFaceAnalysis initialized successfully!")