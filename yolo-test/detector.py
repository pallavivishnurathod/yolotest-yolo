"""
Person Detection Module
Refactored from detections.ipynb into an object-oriented architecture.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
from urllib.request import urlretrieve

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectionConfig:
    """Configuration parameters for detection (primitive attributes only)."""
    conf_threshold: float = 0.25
    image_size: int = 640
    target_class_id: int = 0  # 0 represents 'person' in COCO dataset
    device: str = "cpu"


@dataclass
class BoundingBox:
    """Represents a detected object's bounding box and metadata."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def width(self) -> float:
        """Calculate width of bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Calculate height of bounding box."""
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        """Calculate center X coordinate."""
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        """Calculate center Y coordinate."""
        return (self.y1 + self.y2) / 2.0

    def to_dict(self) -> dict:
        """Serialize bounding box metrics into a dictionary."""
        return {
            "class_id": self.class_id,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y,
        }


class ModelManager:
    """Manages model downloading, caching, and loading."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        model_url: str = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        storage_dir: Optional[Union[str, Path]] = None,
    ):
        self.model_name = model_name
        self.model_url = model_url
        self.storage_dir = (
            Path(storage_dir) if storage_dir else Path.home() / ".files" / "models"
        )
        self.model_path = self.storage_dir / self.model_name

    def ensure_model_exists(self) -> Path:
        """Download model if not present in the local cache."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists():
            print(f"Downloading {self.model_name} from {self.model_url}...")
            urlretrieve(self.model_url, self.model_path)
            print(f"Model downloaded successfully to: {self.model_path}")
        else:
            print(f"Using cached model at: {self.model_path}")
        return self.model_path

    def load_model(self) -> YOLO:
        """Ensure model is available locally and load it into Ultralytics YOLO."""
        model_path = self.ensure_model_exists()
        return YOLO(str(model_path))


class DetectionResult:
    """Encapsulates detection results for an image."""

    def __init__(
        self,
        image_path: Union[str, Path],
        boxes: List[BoundingBox],
        annotated_image: Optional[np.ndarray] = None,
    ):
        self.image_path = Path(image_path)
        self.boxes: List[BoundingBox] = boxes  # Outer relationship to BoundingBox
        self.annotated_image: Optional[np.ndarray] = annotated_image

    def print_summary(self) -> None:
        """Print formatted detection statistics for all detected boxes."""
        print(f"Found {len(self.boxes)} detection(s) in {self.image_path.name}:")
        for idx, box in enumerate(self.boxes, start=1):
            print(f"--- Person {idx} ---")
            print(f"Class ID: {box.class_id}")
            print(f"Confidence: {box.confidence:.4f}")
            print(f"BBox: X1={box.x1:.2f}, Y1={box.y1:.2f}, X2={box.x2:.2f}, Y2={box.y2:.2f}")
            print(f"Dimensions: Width={box.width:.2f}, Height={box.height:.2f}")
            print(f"Center: ({box.center_x:.2f}, {box.center_y:.2f})")

    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save the annotated image to target output directory."""
        if self.annotated_image is None:
            raise ValueError("No annotated image available to save.")
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / self.image_path.name
        cv2.imwrite(str(out_file), self.annotated_image)
        print(f"Annotated result saved to: {out_file}")
        return out_file

    def display(self, title: str = "YOLO11n Person Detection") -> None:
        """Display the annotated image using matplotlib."""
        if self.annotated_image is None:
            raise ValueError("No annotated image available to display.")
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(self.annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(title)
        plt.show()


class PersonDetector:
    """High-level detector coordinating model inference and result formatting."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        model_manager: Optional[ModelManager] = None,
    ):
        # Outer relationships
        self.config: DetectionConfig = config if config is not None else DetectionConfig()
        self.model_manager: ModelManager = model_manager if model_manager is not None else ModelManager()
        self.model: Optional[YOLO] = None

    def load(self) -> None:
        """Load YOLO model if not already loaded."""
        if self.model is None:
            self.model = self.model_manager.load_model()

    def detect(self, image_path: Union[str, Path]) -> DetectionResult:
        """Run person detection on a single image."""
        self.load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        predictions = self.model.predict(
            source=str(path),
            classes=[self.config.target_class_id],
            conf=self.config.conf_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
        )

        detected_boxes: List[BoundingBox] = []
        annotated_img: Optional[np.ndarray] = None

        if predictions:
            first_pred = predictions[0]
            annotated_img = first_pred.plot()

            for box in first_pred.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())

                detected_boxes.append(
                    BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf,
                        class_id=cls_id,
                    )
                )

        return DetectionResult(
            image_path=path,
            boxes=detected_boxes,
            annotated_image=annotated_img,
        )


if __name__ == "__main__":
    # Example execution mimicking detections.ipynb
    project_dir = Path.cwd().parent if Path.cwd().name == "yolo-test" else Path.cwd()
    input_img = project_dir / "dataset" / "images" / "construction_worker.jpg"
    output_directory = project_dir / "runs" / "person_detection"

    detector = PersonDetector()
    result = detector.detect(input_img)
    result.print_summary()
    result.save(output_directory)

