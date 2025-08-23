# Batch Image Labeling

A FastAPI-based application that detects fish in images and returns COCO format annotations using a pre-trained PyTorch model.

## Features

- Fish detection in images via URL
- COCO format annotation output
- RESTful API with health checks
- Docker containerization
- Automated testing

## Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local development)

## Implementation

### Using Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/buzzbing/batch-image-labeling
   cd batch-image-labeling
   git-lfs pull
   ```

2. **Run the application**
   ```bash
   docker-compose up -d
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Health check: http://localhost:8000/health
   - API docs: http://localhost:8000/docs

### Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Run the application**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Usage

### Detect Fish in one Image

```bash
curl -X POST "http://localhost:8000/get-coco-single" \
     -H "Content-Type: application/json" \
     -d '{
       "image_url": "https://example.com/fish-image.jpg",
       "image_id": "fish_001"
     }'
```

### Response Format

Returns COCO format annotations:
```json
{
  "coco": {
    "annotations": [...],
    "images": [...],
    "categories": [...]
  }
}
```

### Detect in Batch

```bash
curl -X POST "http://localhost:8000/get-coco-batch" \
     -H "Content-Type: application/json" \
     -d '{
      "images": [
         {
            "image_url": "https://example.com/fish-image1.jpg",
            "image_id": "fish_001"
         },
         {
            "image_url": "https://example.com/fish-image2.jpg",
            "image_id": "fish_002"
         }
  ]
}'
```

### Response Format

Returns COCO format annotations:
```
[
  {
    "coco": {
      "annotations": [...],
      "images": [...],
      "categories": [...]
    }
   }
  
]
```

### Automatic Image Saving

**Every API request automatically saves the annotated image to the `output/` folder:**
- Images are saved with bounding boxes drawn around detected fish

## Project Structure

```bash
├── Dockerfile
├── README.md
├── app.py
├── docker-compose.yml
├── inference
│   ├── __init__.py
│   ├── class_mapping.json
│   ├── coco_template.json
│   ├── models
│   │   └── model.pt
│   ├── predictor.py
│   └── transforms.json
├── output
├── pyproject.toml
├── requirements.txt
├── test.ipynb
└── tests
    ├── __init__.py
    ├── test_images
    │   └── 800px-Madeira_Fish.jpeg
    └── test_predictor.py
```