import os
import torch
import torchvision
import json
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
from io import BytesIO

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
TEMPLATE_PATH = "inference/coco_template.json"
TRANSFORM_PATH = "inference/transforms.json"


class FishDetector:
    def __init__(self, class_mapping_path: str, modelpath: str):
        self.class_mapping_path = class_mapping_path
        self.modelpath = modelpath
        self.class_mppping = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        with open(self.class_mapping_path) as data:
            mappings = json.load(data)
        self.class_mapping = {
            item["model_idx"]: item["class_name"] for item in mappings
        }
        self.model = torch.jit.load(self.modelpath).to(self.device)

    def get_image(self, image_path: str):
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")
            image = Image.open(image_path).convert("RGB")

        return image

    def transform_image(
        self,
        image_path: str,
        configpath: str = TRANSFORM_PATH,
    ) -> Image:
        image = self.get_image(image_path)
        transforms = A.load(configpath)
        image_np = np.array(image)
        image = transforms(image=image_np)["image"]
        return image

    def predict(self, image_path: str) -> list:
        self.load_model()
        image = self.transform_image(image_path)
        x = torch.from_numpy(image).to(self.device)
        with torch.no_grad():
            x = x.permute(2, 0, 1).float()
            y = self.model(x)
            pred_boxes = y["pred_boxes"]
            pred_classes = y["pred_classes"]
            scores = y["scores"]

            # Score thresholding
            keep = scores > SCORE_THRESHOLD
            pred_boxes = pred_boxes[keep]
            pred_classes = pred_classes[keep]
            scores = scores[keep]

            # IOU thresholding
            to_keep = torchvision.ops.nms(pred_boxes, scores, IOU_THRESHOLD)
            pred_boxes = pred_boxes[to_keep]
            pred_classes = pred_classes[to_keep]
            scores = scores[to_keep]

            results = []
            for bbox, class_idx, score in zip(pred_boxes, pred_classes, scores):
                x1, y1, x2, y2 = bbox.tolist()
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                category_id = int(class_idx)
                category = self.class_mapping.get(category_id, str(category_id))
                results.append(
                    {
                        "bbox": coco_bbox,
                        "score": float(score),
                        "category_id": category_id,
                        "category_name": category,
                    }
                )

            return results

    def draw_boxes_on_image(self, image_path: str, coco_predictions: dict):
        image = self.transform_image(image_path)
        image_draw = np.array(image).copy()
        for pred in coco_predictions:
            x, y, w, h = map(int, pred["bbox"])
            class_name = pred["category_name"]
            score = pred["score"]
            # Draw rectangle
            cv2.rectangle(image_draw, (x, y), (x + w, y + h), (255, 0, 0), 4)
            # Draw label and score
            label = f"{class_name}: {score:.2f}"
            cv2.putText(
                image_draw,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
        # return Image.fromarray(image_draw)
        plt.imshow(image_draw)
        plt.show()

    def get_coco_annotation(self, image_path, predicted_boxes):
        """
        Generates a COCO format dictionary for an image and its annotations.
        """

        image = self.get_image(image_path)
        width, height = image.size

        with open(TEMPLATE_PATH) as file:
            coco_output = json.load(file)

        for k, v in self.class_mapping.items():

            coco_output["categories"].append(
                {
                    "id": k,
                    "name": v,
                    "supercategory": v,
                }
            )

        image_id = 1
        coco_output["images"].append(
            {
                "id": image_id,
                "license": 1,
                "file_name": image_path,
                "height": height,
                "width": width,
            }
        )

        annotation_id_counter = 1

        for pred in predicted_boxes:
            bbox_coords = pred["bbox"]
            x1, y1, x2, y2 = bbox_coords

            coco_x = x1
            coco_y = y1
            coco_width = x2 - x1
            coco_height = y2 - y1

            area = coco_width * coco_height

            coco_output["annotations"].append(
                {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": pred["category_id"],
                    "bbox": [coco_x, coco_y, coco_width, coco_height],
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )

            annotation_id_counter += 1

        return coco_output


# if __name__ == "__main__":
#     class_mapping_path = "inference/class_mapping.json"
#     modelpath = "inference/models/model.pt"
#     predictor = FishDetector(
#         class_mapping_path=class_mapping_path,
#         modelpath=modelpath,
#     )
#     image_path = "YOUR_IMAGE_LINK OR YOUR_IMAGE_PATH"
#     preds = predictor.predict(image_path=image_path)
#     coco_annotation = predictor.get_coco_annotation(
#         image_path=image_path, predicted_boxes=preds
#     )
#     with open("output1.json", "w") as f:
#         json.dump(coco_annotation, f, indent=4)