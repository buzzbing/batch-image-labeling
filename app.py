from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, AnyHttpUrl, Field
from typing import Any, List
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

class BatchImageInput(BaseModel):
    images: List[ImageInput] = Field(..., min_length=1, max_length=5)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fish-detector"}


@app.get("/")
def root():
    return {"message": "Fish Detection API", "version": "1.0.0"}


@app.post("/get-coco-single", response_model=InferenceResponse)
def predict_coco_single(input: ImageInput):
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



@app.post("/get-coco-batch", response_model=List[InferenceResponse])
def predict_batch(input: BatchImageInput, background_tasks: BackgroundTasks):
    responses = []

    for image in input.images:
        try:
            url = str(image.image_url)
            preds = detector.predict(url)
            coco = detector.get_coco_annotation(url, preds, image.image_id)
            background_tasks.add_task(detector.draw_boxes_on_image, url, preds)
            responses.append({"coco": coco})

        except Exception as e:
            responses.append({
                "coco": {
                    "error": str(e),
                    "image_id": image.image_id,
                    "annotations": [],
                    "images": [],
                    "categories": []
                }
            })

    return responses