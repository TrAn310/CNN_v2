import numpy as np
from pathlib import Path

p = Path("FaceRecognition/evaluation_results")

for filename in ["verification_embeddings.npz", "unknown_embeddings.npz"]:
    data = np.load(p / filename)

    print("\nFILE:", filename)

    for key in data.files:
        arr = data[key]
        print(key, arr.shape, arr.dtype)
