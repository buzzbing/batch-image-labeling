from fastapi import FastAPI
from pydantic import BaseModel, AnyHttpUrl
from typing import Any
from inference.predictor import FishDetector

app = FastAPI()

CLASS_MAPPING_PATH = "inference/class_mapping.json"
MODEL_PATH = "inference/models/model.pt"

detector = FishDetector(CLASS_MAPPING_PATH, MODEL_PATH)


class ImageInput(BaseModel):
    image_url: AnyHttpUrl
    image_id: str


class InferenceResponse(BaseModel):
    coco: dict[str, Any]



@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fish-detector"}


@app.get("/")
def root():
    return {"message": "Fish Detection API", "version": "1.0.0"}


@app.post("/get-coco-annotation", response_model=InferenceResponse)
def predict(input: ImageInput):
    if input.image_url:
        try:
            url = str(input.image_url)
            preds = detector.predict(url)
            coco = detector.get_coco_annotation(
                url,
                preds,
                input.image_id,
            )
            detector.draw_boxes_on_image(url, preds)
            return {"coco": coco}
            
        except Exception as e:
            error_coco = {
                "error": str(e),
                "image_id": input.image_id,
                "annotations": [],
                "images": [],
                "categories": []
            }
            return {"coco": error_coco}

    error_coco = {
        "error": "No image URL provided",
        "image_id": input.image_id,
        "annotations": [],
        "images": [],
        "categories": []
    }
    return {"coco": error_coco}
