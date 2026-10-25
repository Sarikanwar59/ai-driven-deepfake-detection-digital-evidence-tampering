import io
import json
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image

MODEL_PATH = Path("models/hog_linear_svm.joblib")
BENCHMARK_ZIP = Path(
    "data/external/faceforensics_benchmark/"
    "faceforensics_benchmark_images.zip"
)
OUTPUT_PATH = Path("reports/faceforensics_benchmark_predictions.json")

IMAGE_SIZE = (64, 64)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def make_hog():
    return cv2.HOGDescriptor(
        (64, 64),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    )


def extract_hog(archive, member, hog):
    raw = archive.read(member)

    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB").resize(IMAGE_SIZE)
        rgb = np.asarray(img, dtype=np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return hog.compute(gray).reshape(-1).astype(np.float32)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not BENCHMARK_ZIP.is_file():
        raise FileNotFoundError(f"Benchmark ZIP not found: {BENCHMARK_ZIP}")

    model = joblib.load(MODEL_PATH)
    hog = make_hog()

    predictions = {}
    errors = []

    with zipfile.ZipFile(BENCHMARK_ZIP, "r") as archive:
        members = sorted(
            name for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in EXTENSIONS
        )

        if not members:
            raise RuntimeError("No benchmark images found in ZIP.")

        for index, member in enumerate(members, start=1):
            filename = Path(member).name

            try:
                features = extract_hog(archive, member, hog)
                label_id = int(model.predict(features.reshape(1, -1))[0])

                if label_id not in (0, 1):
                    raise ValueError(f"Unexpected model label: {label_id}")

                predictions[filename] = "fake" if label_id == 1 else "real"

            except Exception as exc:
                errors.append({
                    "filename": filename,
                    "error": str(exc),
                })

            if index % 100 == 0 or index == len(members):
                print(f"Processed {index}/{len(members)} images")

    if errors:
        print(f"\nERROR: {len(errors)} image(s) failed.")
        for error in errors[:10]:
            print(f"  {error['filename']}: {error['error']}")
        print("No submission JSON written because processing was incomplete.")
        return

    if len(predictions) != len(members):
        raise RuntimeError("Prediction count does not match image count.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print("\n========== BENCHMARK PREDICTIONS ==========")
    print(f"Images processed : {len(members)}")
    print(f"Predictions saved: {len(predictions)}")
    print(f"Model loaded     : {MODEL_PATH}")
    print(f"Output JSON      : {OUTPUT_PATH}")
    print("\nPredicted label counts:")
    print(f"  Real: {sum(v == 'real' for v in predictions.values())}")
    print(f"  Fake: {sum(v == 'fake' for v in predictions.values())}")
    print("\nNo training or model tuning was performed.")


if __name__ == "__main__":
    main()
