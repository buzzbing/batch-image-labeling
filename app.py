from fastapi import FastAPI
from pydantic import BaseModel, AnyHttpUrl
from inference.predictor import FishDetector

app = FastAPI()

CLASS_MAPPING_PATH = "inference/class_mapping.json"
MODEL_PATH = "inference/models/model.pt"

detector = FishDetector(CLASS_MAPPING_PATH, MODEL_PATH)

class ImageInput(BaseModel):
    image_url: AnyHttpUrl
    image_id: str


class InferenceResponse(BaseModel):
    coco: dict


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
            url = str(str(input.image_url))
            preds = detector.predict(url)
            coco = detector.get_coco_annotation(url, preds, input.image_id,)
        except Exception as e:
            raise e
                # return {'coco': f'ERROR {e}'}

        return {"coco": coco}
