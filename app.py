from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from inference.predictor import FishDetector
from typing import Union, List

app = FastAPI()

CLASS_MAPPING_PATH = "inference/class_mapping.json"
MODEL_PATH = "inference/models/model.pt"

detector = FishDetector(CLASS_MAPPING_PATH, MODEL_PATH)

class ImagePath(BaseModel):
    image_path: Union[str, List]


class InferenceResponse(BaseModel):
    coco: dict


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fish-detector"}


@app.get("/")
def root():
    return {"message": "Fish Detection API", "version": "0.1.0"}


@app.post("/get-coco-annotation", response_model=InferenceResponse)
def predict(input: ImagePath):
    if isinstance(input.image_path, List):
        results = []
        for image in input.image_path:
            try:
                preds = detector.predict(image)
                coco = detector.get_coco_annotation(image, preds)
                results.append(coco)
            except Exception as e:
                return JSONResponse({'results': f'ERROR {e}'})
        return JSONResponse({"results": results})

    if isinstance(input.image_path, str):
        try:
            preds = detector.predict(input.image_path)
            coco = detector.get_coco_annotation(input.image_path, preds)
        except Exception as e:
                return JSONResponse({'results': f'ERROR {e}'})
        return JSONResponse({"results": coco})
