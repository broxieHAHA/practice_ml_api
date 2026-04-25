import logging
import os
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("iris_api")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "iris_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Iris Classification API",
    description="Классификация ирисов",
    version="2.0.0",
)

logger.info(f"Загрузка модели: models:/{MODEL_NAME}/{MODEL_STAGE}")
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    model = None

SPECIES = ["setosa", "versicolor", "virginica"]

class Features(BaseModel):
    slength: float
    swidth: float
    plength: float
    pwidth: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "slength": 5.1,
                "swidth": 3.5,
                "plength": 1.4,
                "pwidth": 0.2
            }
        }
    }

class Result(BaseModel):
    species: str
    class_id: int
    model_name: str
    model_stage: str


@app.get("/health")
async def health():
    logger.info("Health check")
    return {"status": "OK", "model_loaded": model is not None}


@app.post("/predict", response_model=Result)
async def predict(features: Features):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    logger.info(f"Запрос: sl={features.slength}, sw={features.swidth}, "
                f"pl={features.plength}, pw={features.pwidth}")

    try:
        df = pd.DataFrame([[features.slength, features.swidth,
                            features.plength, features.pwidth]])
        prediction = int(model.predict(df)[0])
        species_name = SPECIES[prediction]

        logger.info(f"Результат: {species_name} (class={prediction})")
        return Result(
            species=species_name,
            class_id=prediction,
            model_name=MODEL_NAME,
            model_stage=MODEL_STAGE,
        )
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)