"""Download the MediaPipe pose model required by live workout tracking."""

from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).resolve().parents[1] / "coachvision" / "ai" / "pose_landmarker.task"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        print(f"Pose model already exists at {MODEL_PATH}")
        return
    print(f"Downloading pose model to {MODEL_PATH}")
    urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Downloaded {MODEL_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
