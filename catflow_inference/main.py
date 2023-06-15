from fastapi import FastAPI, Body
from starlette.responses import StreamingResponse
from PIL import Image
import numpy as np
import io
import torch
import os
import cv2


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
            MODEL = torch.hub.load("ultralytics/yolov5", "yolov5m")
        else:
            # Check for model weights
            model_weights = os.getenv("YOLO_WEIGHTS")
            if model_weights is None:
                raise ValueError("YOLO_WEIGHTS environment variable is not set.")
            elif not os.path.isfile(model_weights):
                raise FileNotFoundError(
                    f"YOLO_WEIGHTS file '{model_weights}' not found."
                )

            MODEL = torch.hub.load("ultralytics/yolov5", "custom", path=model_weights)

        # Set global threshold
        THRESHOLD = threshold


def centerxywh_to_xyxy(box):
    x, y, width, height = box
    start_x = int(x - width / 2)
    start_y = int(y - height / 2)
    end_x = int(x + width)
    end_y = int(y + height)

    return start_x, start_y, end_x, end_y


def get_predictions(model, image, threshold):
    results = model(image)
    predictions = results.xywh[0]

    confidences = []
    labels = []
    boxes = []
    for prediction in predictions:
        x, y, width, height, confidence, class_id = prediction
        if confidence < threshold:
            continue

        labels.append(results.names[class_id.item()])
        confidences.append(confidence.item())
        boxes.append([x.item(), y.item(), width.item(), height.item()])

    return boxes, confidences, labels


@app.post("/predict/")
async def predict(body: bytes = Body(...)):
    load_config()
    image = Image.open(io.BytesIO(body))
    boxes, confidences, labels = get_predictions(MODEL, image, THRESHOLD)
    return {"boxes": boxes, "confidences": confidences, "classes": labels}


@app.post("/predict/draw/")
async def predict_draw(body: bytes = Body(...)):
    load_config()
    image = Image.open(io.BytesIO(body))
    boxes, confidences, labels = get_predictions(MODEL, image, THRESHOLD)

    # Convert PIL Image to OpenCV format (numpy array) and get dimensions
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]

    # Draw the bounding boxes
    for box in boxes:
        start_x, start_y, end_x, end_y = centerxywh_to_xyxy(box)
        image = cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

    # Convert the image back to bytes
    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
