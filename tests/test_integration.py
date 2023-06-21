from starlette.testclient import TestClient
from PIL import Image
import io
import os
import pytest
import json
import base64
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
    assert len(predictions) == 1
    assert predictions[0]["label"] == "car"


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


def test_motion_endpoint():
    """Test that the /motion/ endpoint detects the car in this video and returns
    appropriate metadata"""
    with open("tests/test_images/car.mp4", "rb") as video_file:
        file_data = {"file": video_file}
        class_data = {"classes": json.dumps(["car", "cat"])}
        response = client.post("/motion/", files=file_data, data=class_data)
        assert response.status_code == 200

    data = json.loads(response.content)
    assert "detections" in data
    assert "label" in data
    assert "image" in data

    print(f"Found {data['detections']} instances of {data['label']}")
    assert data["detections"] == 11
    assert data["label"] == "car"

    content = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(content))
    assert image.format == "PNG"
    assert image.size == (640, 480)


def test_motion_endpoint_class_filter():
    """As above, but we should get no detections as we're not looking for cars"""
    with open("tests/test_images/car.mp4", "rb") as video_file:
        file_data = {"file": video_file}
        class_data = {"classes": json.dumps(["cat", "capybara"])}
        response = client.post("/motion/", files=file_data, data=class_data)
        assert response.status_code == 200

    data = json.loads(response.content)
    assert "detections" in data
    assert "label" in data
    assert "image" in data

    assert data["detections"] == 0
    assert data["label"] == ""
