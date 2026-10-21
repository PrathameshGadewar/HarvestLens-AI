import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
from backend.app.schemas import PredictResponse
from backend.app.ml.detector import FruitDetector
from backend.app.config import ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE_MB

logger = logging.getLogger("harvestlens.predict")
router = APIRouter(tags=["AI Inference"])


@router.get("/health")
@router.get("/api/health")
def health_check():
    """Health check endpoint to verify backend status and model readiness."""
    try:
        detector = FruitDetector.get_instance()
        class_count = len(detector.names)
        return {
            "status": "healthy",
            "model_ready": True,
            "classes_count": class_count,
            "classes": detector.names
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "model_ready": False,
            "error": str(e)
        }


@router.post("/predict", response_model=PredictResponse)
@router.post("/api/predict", response_model=PredictResponse)
async def predict_fruit(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(None, ge=0.01, le=1.0, description="Confidence threshold (0.01 to 1.0)")
):
    """
    Execute real YOLOv8 fruit maturity inference on the uploaded image.
    Recognizes 32 fruit-maturity classes across 8 fruits and 4 maturity stages.
    """
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image file provided.")

    # Validate content type or extension
    filename = file.filename or "upload.jpg"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext and ext not in ALLOWED_EXTENSIONS and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed image extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        content = await file.read()
        if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image file exceeds maximum allowed size of {MAX_IMAGE_SIZE_MB} MB."
            )

        detector = FruitDetector.get_instance()
        result = detector.predict(content, confidence_threshold=conf)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI model inference failed: {str(e)}"
        )
