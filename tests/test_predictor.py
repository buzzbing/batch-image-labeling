import pytest
import numpy as np
from unittest import mock
import os
import json
import torch
from inference import FishDetector


@pytest.fixture
def class_mapping_path(tmp_path):
    """Create temporary class mapping JSON file for testing.

    Args:
        tmp_path: pytest fixture providing temporary directory path

    Returns:
        str: Path to temporary class mapping JSON file
    """
    path = tmp_path / "class_mapping.json"
    with open(path, "w") as f:
        json.dump([{"model_idx": 0, "class_name": "Fish"}], f)
    return str(path)


@pytest.fixture
def model_path(tmp_path):
    """Create temporary dummy model file for testing.

    Args:
        tmp_path: pytest fixture providing temporary directory path

    Returns:
        str: Path to temporary dummy model file
    """
    path = tmp_path / "model.pt"
    with open(path, "wb") as f:
        f.write(b"dummy")
    return str(path)


@pytest.fixture
def test_images_dir():
    """Fixture to get the path to test images directory.

    Returns:
        str: Absolute path to test images directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "test_images")


@pytest.fixture
def sample_image_path(test_images_dir):
    """Fixture to get a sample image path from test_images.

    Args:
        test_images_dir: pytest fixture providing test images directory path

    Returns:
        str: Path to sample test image file
    """
    return os.path.join(test_images_dir, "800px-Madeira_Fish.jpeg")


@pytest.fixture
def transforms_path(tmp_path):
    """Create temporary transforms configuration JSON file for testing.

    Args:
        tmp_path: pytest fixture providing temporary directory path

    Returns:
        str: Path to temporary transforms configuration file
    """
    path = tmp_path / "transforms.json"
    transform_config = {
        "__version__": "1.1.0",
        "transform": {
            "__class_fullname__": "Compose",
            "p": 1.0,
            "transforms": [],
            "bbox_params": None,
            "keypoint_params": None,
            "additional_targets": {},
        },
    }
    with open(path, "w") as f:
        json.dump(transform_config, f)
    return str(path)


@mock.patch("inference.predictor.torch.jit.load")
def test_fish_detector_initialization(
    mock_jit_load,
    class_mapping_path,
    model_path,
):
    """Test FishDetector initialization with mocked PyTorch model.

    Args:
        mock_jit_load: Mocked torch.jit.load function
        class_mapping_path: Path to class mapping JSON file
        model_path: Path to model file

    Returns:
        None
    """
    mock_model = mock.Mock()
    mock_model.to.return_value = mock_model
    mock_jit_load.return_value = mock_model
    detector = FishDetector(class_mapping_path, model_path)
    mock_jit_load.assert_called_once_with(model_path)
    mock_model.to.assert_called_once()
    assert detector.class_mapping[0] == "Fish"
    assert detector.model == mock_model


def test_transform_image(class_mapping_path, model_path, sample_image_path):
    """Test image transformation with size and ratio validation.

    Args:
        class_mapping_path: Path to class mapping JSON file
        model_path: Path to model file
        sample_image_path: Path to sample test image

    Returns:
        None
    """
    with mock.patch("inference.predictor.torch.jit.load") as mock_jit_load:
        mock_model = mock.Mock()
        mock_model.to.return_value = mock_model
        mock_jit_load.return_value = mock_model
        
        detector = FishDetector(class_mapping_path, model_path)

        original_image = detector.get_image(sample_image_path)
        original_width, original_height = original_image.size

        result = detector.transform_image(sample_image_path)

        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

        transformed_height, transformed_width, channels = result.shape

        max_dimension = max(transformed_width, transformed_height)
        assert max_dimension <= 1333

        original_ratio = original_width / original_height
        transformed_ratio = transformed_width / transformed_height
        ratio_diff = abs(original_ratio - transformed_ratio)
        assert ratio_diff < 0.1


@mock.patch("inference.predictor.torch.jit.load")
def test_predict(
    mock_jit_load,
    class_mapping_path,
    model_path,
    sample_image_path,
):
    """Test prediction functionality with mocked model outputs.

    Args:
        mock_jit_load: Mocked torch.jit.load function
        class_mapping_path: Path to class mapping JSON file
        model_path: Path to model file
        sample_image_path: Path to sample test image

    Returns:
        None
    """
    mock_model = mock.Mock()
    mock_model.to.return_value = mock_model
    mock_jit_load.return_value = mock_model
    detector = FishDetector(class_mapping_path, model_path)
    assert detector.class_mapping[0] == "Fish"

    mock_model.return_value = {
        "pred_boxes": torch.tensor([[10, 10, 50, 50]]),
        "pred_classes": torch.tensor([0]),
        "scores": torch.tensor([0.9]),
    }

    with mock.patch(
        "inference.predictor.torchvision.ops.nms",
        return_value=torch.tensor([0]),
    ):
        results = detector.predict(sample_image_path)

    assert isinstance(results, list)
    assert results[0]["category_name"] == "Fish"


@mock.patch("inference.predictor.Image.open")
def test_get_coco_annotation(
    mock_image_open,
    class_mapping_path,
    model_path,
    sample_image_path,
):
    """Test COCO annotation generation with mocked dependencies.

    Args:
        mock_image_open: Mocked PIL.Image.open function
        class_mapping_path: Path to class mapping JSON file
        model_path: Path to model file
        sample_image_path: Path to sample test image

    Returns:
        None
    """
    with mock.patch("inference.predictor.torch.jit.load") as mock_jit_load:
        mock_model = mock.Mock()
        mock_model.to.return_value = mock_model
        mock_jit_load.return_value = mock_model
        
        detector = FishDetector(class_mapping_path, model_path)
        assert detector.class_mapping[0] == "Fish"
        
        mock_image = mock.Mock()
        mock_image.size = (100, 100)
        mock_image_open.return_value = mock_image
        
        predicted_boxes = [
            {
                "bbox": [10, 10, 40, 40],
                "score": 0.9,
                "category_id": 0,
                "category_name": "Fish",
            }
        ]

        with mock.patch.object(detector, "get_image", return_value=mock_image):
            result = detector.get_coco_annotation(
                sample_image_path,
                predicted_boxes,
                1,
            )
        assert "images" in result
        assert "annotations" in result
        assert result["annotations"][0]["category_id"] == 0



def test_model_loading_in_init(class_mapping_path, model_path):
    """Test that model loading happens during initialization.
    
    Args:
        class_mapping_path: Path to class mapping JSON file
        model_path: Path to model file
        
    Returns:
        None
    """
    with mock.patch("inference.predictor.torch.jit.load") as mock_jit_load:
        mock_model = mock.Mock()
        mock_model.to.return_value = mock_model
        mock_jit_load.return_value = mock_model
        
        detector = FishDetector(class_mapping_path, model_path)
        mock_jit_load.assert_called_once_with(model_path)
        mock_model.to.assert_called_once()
        assert detector.model == mock_model