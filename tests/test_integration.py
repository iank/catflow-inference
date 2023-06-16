from starlette.testclient import TestClient
from PIL import Image
import io
import os
import pytest
from catflow_inference.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_environment():
    os.environ["YOLO_PRETRAINED"] = "1"
    os.environ["YOLO_THRESHOLD"] = "0.5"

    yield

    del os.environ["YOLO_PRETRAINED"]
    del os.environ["YOLO_THRESHOLD"]


def test_status_endpoint():
    response = client.get("/status/")
    assert response.status_code == 200

    status = response.json()
    assert status["model"] == "pretrained"
    print(len(status["classes"]))
    assert len(status["classes"]) == 80


def test_predict_endpoint():
    # Prepare the test image
    with open("tests/test_images/wikipedia_cc_640x480.png", "rb") as image_file:
        image_bytes = image_file.read()

    response = client.post(
        "/predict/",
        headers={"Content-Type": "application/octet-stream"},
        content=image_bytes,
    )
    assert response.status_code == 200

    predictions = response.json()
    assert "boxes" in predictions
    assert "confidences" in predictions
    assert "classes" in predictions

    assert len(predictions["classes"]) == 1
    assert "car" in predictions["classes"]


def test_predict_draw_endpoint():
    # Prepare the test image
    with open("tests/test_images/wikipedia_cc_640x480.png", "rb") as image_file:
        image_bytes = image_file.read()

    response = client.post(
        "/predict/draw/",
        headers={"Content-Type": "application/octet-stream"},
        content=image_bytes,
    )
    assert response.status_code == 200

    # Check if a valid PNG image is returned
    image_stream = io.BytesIO(response.content)
    image = Image.open(image_stream)
    assert image.format == "PNG"
