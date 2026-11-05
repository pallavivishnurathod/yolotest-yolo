from pathlib import Path
from ultralytics import YOLO


HOME_DIR = Path.home()
MODEL_DIR = HOME_DIR / ".files" / "models"


def load_model(model_name="yolo11n.pt"):

    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = YOLO(str(model_path))

    print("YOLO model loaded successfully!")

    return model