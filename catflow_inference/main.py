from fastapi import FastAPI, Body, HTTPException, UploadFile, Form, File
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from PIL import Image
from frameextractor.utils.image_processing import extract_frames
from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np
import base64
import tempfile
import io
import aiofiles
import torch
import os
import cv2
import logging
import json
import scipy

logger = logging.getLogger(__name__)


app = FastAPI()

MODEL = None
THRESHOLD = None


def load_config():
    global MODEL
    global THRESHOLD

    if MODEL is None:
        # Check for and load threshold value
        threshold_str = os.getenv("YOLO_THRESHOLD")
        if threshold_str is None:
            threshold = 0.5
        else:
            try:
                threshold = float(threshold_str)
            except ValueError as e:
                raise ValueError("YOLO_THRESHOLD must be a float.") from e
            if not 0 <= threshold <= 1:
                raise ValueError("YOLO_THRESHOLD must be between 0 and 1.")

        # Load model
        pretrained_flag = os.getenv("YOLO_PRETRAINED")
        if pretrained_flag == "1":
            MODEL = torch.hub.load("ultralytics/yolov5", "yolov5m", trust_repo=True)
            MODEL.catflow_name = "pretrained"
        else:
            # Check for model weights
            model_weights = os.getenv("YOLO_WEIGHTS")
            if model_weights is None:
                raise ValueError("YOLO_WEIGHTS environment variable is not set.")
            elif not os.path.isfile(model_weights):
                raise FileNotFoundError(
                    f"YOLO_WEIGHTS file '{model_weights}' not found."
                )

            MODEL = torch.hub.load(
                "ultralytics/yolov5", "custom", path=model_weights, trust_repo=True
            )
            MODEL.catflow_name = model_weights

        # Set global threshold
        THRESHOLD = threshold


def centerxywh_to_xyxy(box):
    x, y, width, height = box
    start_x = int(x - width / 2)
    start_y = int(y - height / 2)
    end_x = int(start_x + width)
    end_y = int(start_y + height)

    return start_x, start_y, end_x, end_y


def get_predictions(model, image, threshold):
    results = model(image)
    predictions = results.xywh[0]

    ret = []
    for prediction in predictions:
        x, y, width, height, confidence, class_id = prediction
        if confidence < threshold:
            continue

        label = results.names[class_id.item()]
        confidence = confidence.item()
        box = [x.item(), y.item(), width.item(), height.item()]
        ret.append({"box": box, "confidence": confidence, "label": label})

    return ret


def get_most_detected_class(
    frames: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    # Find the most detected class
    detections = Counter()
    for frame in frames:
        detected_labels = set([x["label"] for x in frame["detections"]])
        for label in detected_labels:
            detections[label] += 1

    if len(detections) == 0:
        return None, []

    most_detected_class = detections.most_common(1)[0][0]

    # Get the highest confidence detection for the most detected class from each frame
    detected_frames = []
    for frame in frames:
        detections = [
            x for x in frame["detections"] if x["label"] == most_detected_class
        ]
        if len(detections) == 0:
            continue

        most_confident_detection = max(detections, key=lambda x: x["confidence"])
        detected_frames.append(
            {"frame": frame["frame"], "detection": most_confident_detection}
        )

    return most_detected_class, detected_frames


@app.get("/status/")
async def read_status():
    load_config()

    if hasattr(MODEL, "names") and hasattr(MODEL, "catflow_name"):
        return {"model": MODEL.catflow_name, "classes": MODEL.names}
    else:
        raise HTTPException(status_code=500, detail="model not loaded")


@app.post("/predict/")
async def predict(body: bytes = Body(...)):
    load_config()
    image = Image.open(io.BytesIO(body))
    predictions = get_predictions(MODEL, image, THRESHOLD)
    return predictions


def draw_detection(frame, predictions):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    imgmask = np.ones(image.shape[:2])
    y_coords, x_coords = np.ogrid[: image.shape[0], : image.shape[1]]

    # Draw the detections
    for prediction in predictions:
        x, y, width, height = prediction["box"]
        dist_from_center = ((x_coords - x) / width) ** 2 + (
            (y_coords - y) / height
        ) ** 2
        mask = dist_from_center <= 0.5
        imgmask[mask] = 0

    sigma = 20
    imgmask = scipy.ndimage.gaussian_filter(imgmask, sigma)
    imgmask = 1 - imgmask
    imgmask = imgmask * 0.85 + 0.15
    image[:, :, 2] = (image[:, :, 2] * imgmask).astype("uint8")

    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    # Encode
    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()
    return img_bytes


@app.post("/predict/draw/")
async def predict_draw(body: bytes = Body(...)):
    load_config()
    image = Image.open(io.BytesIO(body))
    predictions = get_predictions(MODEL, image, THRESHOLD)

    # Convert PIL Image to OpenCV format (numpy array) and get dimensions
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_bytes = draw_detection(image, predictions)

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


@app.post("/motion/")
async def motion(file: UploadFile = File(...), classes: str = Form(...)):
    load_config()

    # Classes to look for
    classes_of_interest = json.loads(classes)

    # Extract frames from video
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        async with aiofiles.open(tmpfile.name, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        frames = extract_frames(tmpfile.name)

    logger.info(f"Processing {len(frames)} frames for {classes_of_interest}")

    # Process frames
    detected_frames = []
    for frame in frames:
        predictions = get_predictions(MODEL, frame, THRESHOLD)
        predictions = [x for x in predictions if x["label"] in classes_of_interest]
        if len(predictions) > 0:
            detected_frames.append({"frame": frame, "detections": predictions})

    response = {
        "detections": len(detected_frames),
        "label": "",
        "image": "",
    }

    if response["detections"] == 0:
        logger.info("No detections")
        return JSONResponse(content=response)

    # Pick the object we saw the most
    most_detected_class, detected_frames = get_most_detected_class(detected_frames)
    logger.info(f"Detected {len(detected_frames)} instances of {most_detected_class}")
    response["detections"] = len(detected_frames)
    response["label"] = most_detected_class

    # Pick the middle sighting
    pick_frame = detected_frames[len(detected_frames) // 2]
    img_bytes = draw_detection(pick_frame["frame"], [pick_frame["detection"]])

    # Convert the frame to base64 to include in JSON response
    base64_im = base64.b64encode(img_bytes).decode("utf-8")

    response["image"] = base64_im
    return JSONResponse(content=response)
