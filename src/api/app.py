from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import setup_logging, get_logger
from src.exception import ArtifactNotFoundError, PredictionError

# initialize logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Fraud Prediction API")


class Transaction(BaseModel):
    Amount: float
    # MerchantID removed: aggregate mappings used instead
    TransactionType: Optional[str] = "purchase"
    Location: Optional[str] = "New York"
    hour: Optional[int] = None
    day: Optional[int] = None
    weekday: Optional[int] = None
    is_weekend: Optional[int] = None


pipeline: Optional[PredictionPipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    try:
        pipeline = PredictionPipeline()
        logger.info("PredictionPipeline loaded in FastAPI startup")
    except ArtifactNotFoundError as e:
        logger.error(f"Artifacts missing at startup: {e}")
        pipeline = None
    except Exception as e:
        logger.error(f"Failed to initialize PredictionPipeline: {e}")
        pipeline = None


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


@app.post("/predict")
def predict(tx: Transaction):
    if pipeline is None:
        logger.error("Prediction requested but pipeline not initialized")
        raise HTTPException(status_code=503, detail="Model artifacts not available")

    try:
        data = tx.dict()
        result = pipeline.predict_risk(data)
        return result
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
