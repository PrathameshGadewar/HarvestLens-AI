import io
import time
import base64
import logging
from typing import List, Dict, Any, Optional
import cv2
from PIL import Image
from ultralytics import YOLO

from backend.app.config import MODEL_PATH, DEFAULT_CONFIDENCE, FRUITS, MATURITY_STAGES

logger = logging.getLogger("harvestlens.ml")


def parse_fruit_maturity(label: str) -> Dict[str, str]:
    """
    Parse a class label like 'Mango Ripe' into fruit and maturity stage.
    Uses exact token matching first, and sorts candidates by length descending
    to avoid substring confusion (e.g. 'Ripe' matching inside 'Overripe').
    """
    raw = label.replace("_", " ").strip()
    words = [w.capitalize() for w in raw.split()]

    detected_fruit = "Unknown"
    detected_maturity = "Unknown"

    # Exact token matching
    for w in words:
        if w in FRUITS and detected_fruit == "Unknown":
            detected_fruit = w
        if w in MATURITY_STAGES and detected_maturity == "Unknown":
            detected_maturity = w

    # Fallback with length-descending order (Overripe/Unripe checked before Ripe)
    if detected_fruit == "Unknown":
        for f in sorted(FRUITS, key=len, reverse=True):
            if f.lower() in raw.lower():
                detected_fruit = f
                break

    if detected_maturity == "Unknown":
        for m in sorted(MATURITY_STAGES, key=len, reverse=True):
            if m.lower() in raw.lower():
                detected_maturity = m
                break

    if detected_fruit == "Unknown" and len(words) >= 1:
        detected_fruit = words[0]
    if detected_maturity == "Unknown" and len(words) >= 2:
        detected_maturity = words[1]

    return {"fruit": detected_fruit, "maturity": detected_maturity}


class FruitDetector:
    _instance: Optional["FruitDetector"] = None

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(MODEL_PATH)
        logger.info(f"Loading YOLOv8 model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.names = self.model.names
        self.warmup()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "FruitDetector":
        """Singleton accessor to guarantee model is loaded only once at backend startup."""
        if cls._instance is None:
            cls._instance = FruitDetector(model_path)
        return cls._instance

    def warmup(self):
        """Warm up the model with a blank image to avoid inference lag on first user request."""
        try:
            dummy_img = Image.new("RGB", (320, 320), color=(128, 128, 128))
            self.model(dummy_img, conf=0.5, verbose=False)
            logger.info("YOLOv8 model warmup completed successfully.")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def predict(
        self,
        image_bytes: bytes,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run inference on raw image bytes.
        Returns:
            {
                "detections": [
                    {
                        "label": str,
                        "fruit": str,
                        "maturity": str,
                        "confidence": float,
                        "box": [x1, y1, x2, y2],
                        "class_id": int
                    }
                ],
                "image_width": int,
                "image_height": int,
                "inference_time_ms": float
            }
        """
        conf = confidence_threshold if confidence_threshold is not None else DEFAULT_CONFIDENCE

        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size

        start_time = time.perf_counter()
        results = self.model(img, conf=conf, verbose=False)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        detections: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2] in pixels
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                label_name = self.names.get(cls_id, f"Class_{cls_id}")

                parsed = parse_fruit_maturity(label_name)

                detections.append({
                    "label": label_name,
                    "fruit": parsed["fruit"],
                    "maturity": parsed["maturity"],
                    "confidence": round(score, 4),
                    "box": [round(c, 2) for c in xyxy],
                    "class_id": cls_id
                })

        annotated_image = None
        if results and len(results) > 0 and len(detections) > 0:
            try:
                pl = results[0].plot()
                success, buf = cv2.imencode(".jpg", pl)
                if success:
                    annotated_image = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
            except Exception as e:
                logger.warning(f"Failed to generate annotated plot: {e}")

        return {
            "detections": detections,
            "image_width": width,
            "image_height": height,
            "inference_time_ms": inference_time_ms,
            "annotated_image": annotated_image
        }

    @staticmethod
    def calculate_dominant_maturity(detections: List[Dict[str, Any]]) -> Optional[str]:
        """Calculate aggregate dominant maturity based on confidence scores."""
        if not detections:
            return None
        scores: Dict[str, float] = {}
        for d in detections:
            mat = d.get("maturity")
            conf = d.get("confidence", 0.0)
            if mat in MATURITY_STAGES:
                scores[mat] = scores.get(mat, 0.0) + conf
        if not scores:
            return None
        return max(scores.items(), key=lambda x: x[1])[0]
