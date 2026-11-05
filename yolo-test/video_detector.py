"""
Video Person Detection Module
Refactored from video_detection.ipynb into an object-oriented architecture.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Union

import cv2
from ultralytics import YOLO

from detector import BoundingBox, DetectionConfig, ModelManager


@dataclass
class VideoMetadata:
    """Stores video stream metadata (primitives only)."""
    fps: float
    width: int
    height: int
    total_frames: int
    duration_seconds: float


@dataclass
class VideoProcessingSummary:
    """Stores performance and summary metrics of video inference (primitives only)."""
    total_frames_processed: int
    original_fps: float
    average_detection_time: float
    processing_fps: float
    output_path: str


class VideoPersonDetector:
    """High-level video processor managing capture, frame inference, and writing."""

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
        """Load YOLO model if not already in memory."""
        if self.model is None:
            self.model = self.model_manager.load_model()

    def get_video_metadata(self, capture: cv2.VideoCapture) -> VideoMetadata:
        """Extract metadata primitives from an opened VideoCapture stream."""
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 25.0

        duration = (total_frames / fps) if total_frames > 0 else 0.0

        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration_seconds=duration,
        )

    def process_video(
        self,
        input_video_path: Union[str, Path],
        output_video_path: Union[str, Path],
        codec: str = "mp4v",
    ) -> VideoProcessingSummary:
        """Process video frame-by-frame, detect persons, and write annotated video."""
        self.load()

        in_path = Path(input_video_path)
        out_path = Path(output_video_path)

        if not in_path.exists():
            raise FileNotFoundError(f"Unable to open video: {in_path}")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video stream: {in_path}")

        meta = self.get_video_metadata(cap)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            str(out_path),
            fourcc,
            meta.fps,
            (meta.width, meta.height),
        )

        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to initialize output video writer: {out_path}")

        frame_count = 0
        total_detection_time = 0.0

        print(f"Starting video processing: {in_path.name} -> {out_path.name}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                results = self.model.predict(
                    source=frame,
                    classes=[self.config.target_class_id],
                    conf=self.config.conf_threshold,
                    imgsz=self.config.image_size,
                    device=self.config.device,
                    verbose=False,
                )
                detection_time = time.time() - start_time
                total_detection_time += detection_time

                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")
        finally:
            cap.release()
            out.release()

        avg_time = (total_detection_time / frame_count) if frame_count > 0 else 0.0
        avg_fps = (frame_count / total_detection_time) if total_detection_time > 0 else 0.0

        summary = VideoProcessingSummary(
            total_frames_processed=frame_count,
            original_fps=meta.fps,
            average_detection_time=avg_time,
            processing_fps=avg_fps,
            output_path=str(out_path),
        )

        print("\n========== VIDEO PROCESSING COMPLETED ==========")
        print(f"Total Frames Processed: {summary.total_frames_processed}")
        print(f"Original Video FPS: {summary.original_fps:.2f}")
        print(f"Average Detection Time: {summary.average_detection_time:.4f} s")
        print(f"Processing FPS: {summary.processing_fps:.2f}")
        print(f"Saved Output: {summary.output_path}")

        return summary


if __name__ == "__main__":
    project_dir = Path.cwd()
    input_video = project_dir / "dataset" / "video.mp4"
    output_video = project_dir / "runs" / "person_detection_video" / "output.mp4"

    if input_video.exists():
        detector = VideoPersonDetector()
        detector.process_video(input_video, output_video)
    else:
        print(f"Notice: {input_video} not found. Place video.mp4 in dataset/ to run.")

