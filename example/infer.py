import http.client
import cv2
import numpy as np
import argparse
from PIL import Image
from io import BytesIO


def display_image_from_response(response):
    # Load image from HTTP response
    image = Image.open(BytesIO(response))

    # Convert the image to OpenCV format (numpy array) and BGR color format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Display the image
    cv2.imshow("Predicted Image", image)
    cv2.waitKey(0)  # wait until user closes the window
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def post_image(image_path):
    # Prepare headers for http request
    headers = {"Content-Type": "application/octet-stream"}

    # Open image in binary mode
    with open(image_path, "rb") as f:
        img_data = f.read()

    # Send the post request to the server
    conn = http.client.HTTPConnection("localhost:8001")
    conn.request("POST", "/predict/draw/", body=img_data, headers=headers)

    # Get response
    response = conn.getresponse()

    if response.status == 200:
        # Load and display the image from the response
        display_image_from_response(response.read())
    else:
        print("Error:", response.status, response.reason)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send an image to the server and display the predicted image."
    )
    parser.add_argument("image", help="Path to the image file")

    args = parser.parse_args()

    post_image(args.image)
