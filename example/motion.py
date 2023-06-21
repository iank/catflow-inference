import cv2
import requests
import numpy as np
import argparse
from PIL import Image
from io import BytesIO
import json
import base64


def display_image_from_response(response):
    data = json.loads(response)

    print(f"Found {data['detections']} instances of {data['label']}")

    content = base64.b64decode(data["image"])
    # Load image from HTTP response
    image = Image.open(BytesIO(content))

    # Convert the image to OpenCV format (numpy array) and BGR color format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Display the image
    cv2.imshow("Predicted Image", image)
    cv2.waitKey(0)  # wait until user closes the window
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def post_video(video_path):
    # Open video in binary mode
    with open(video_path, "rb") as f:
        file_data = {"file": f}
        keys_data = {"classes": json.dumps(["oscar", "bear"])}

        response = requests.post(
            "https://localhost/inference/motion/",
            files=file_data,
            data=keys_data,
        )

    if response.status_code == 200:
        # Load and display the image from the response
        display_image_from_response(response.content)
    else:
        print("Error:", response.status_code, response.reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send an video to the server and display the detected image"
    )
    parser.add_argument("video", help="Path to the video file")

    args = parser.parse_args()

    post_video(args.video)
