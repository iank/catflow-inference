from catflow_inference.main import centerxywh_to_xyxy
from catflow_inference.main import get_most_detected_class


def test_box_translation():
    # This test validates the bounding box translation
    box = [100.20, 83.83, 53.0, 99]

    start_x, start_y, end_x, end_y = centerxywh_to_xyxy(box)
    assert start_x == 73
    assert start_y == 34
    assert end_x == 126
    assert end_y == 133


def test_get_most_detected_class():
    # We want to get the class that is seen in the most frames, as well as each
    # frame/detection. Two details:
    #
    # If class Y is seen 10 times in 1 frame but 10 times overall, and class X
    # is seen once in 4 frames each, we want class X.
    #
    # If class X is the most-detected-class, and it's seen twice in one frame, just
    # pick the detection w/ the highest confidence for that frame.
    frames = [
        {
            "frame": [1, 0, 0],
            "detections": [
                {"label": "X", "confidence": 0.97, "box": 1},
                {"label": "X", "confidence": 0.50, "box": 2},
            ],
        },
        {
            "frame": [2, 0, 0],
            "detections": [
                {"label": "Y", "confidence": 0.77, "box": 3},
                {"label": "X", "confidence": 0.53, "box": 4},
            ],
        },
        {
            "frame": [3, 0, 0],
            "detections": [
                {"label": "X", "confidence": 0.42, "box": 5},
                {"label": "Z", "confidence": 0.30, "box": 6},
            ],
        },
        {
            "frame": [4, 0, 0],
            "detections": [
                {"label": "X", "confidence": 0.85, "box": 7},
                {"label": "Y", "confidence": 0.23, "box": 8},
                {"label": "Y", "confidence": 0.24, "box": 9},
                {"label": "Y", "confidence": 0.28, "box": 10},
                {"label": "Y", "confidence": 0.39, "box": 11},
                {"label": "Y", "confidence": 0.99, "box": 12},
                {"label": "Y", "confidence": 0.87, "box": 13},
                {"label": "Y", "confidence": 0.20, "box": 14},
            ],
        },
        {
            "frame": [5, 0, 0],
            "detections": [
                {"label": "Z", "confidence": 0.80, "box": 9},
            ],
        },
    ]

    most_detected_class, detected_frames = get_most_detected_class(frames)

    assert most_detected_class == "X"
    assert len(detected_frames) == 4
    assert detected_frames[0]["detection"]["box"] == 1
    assert detected_frames[0]["frame"] == [1, 0, 0]
    assert detected_frames[1]["detection"]["box"] == 4
    assert detected_frames[1]["frame"] == [2, 0, 0]
    assert detected_frames[2]["detection"]["box"] == 5
    assert detected_frames[2]["frame"] == [3, 0, 0]
    assert detected_frames[3]["detection"]["box"] == 7
    assert detected_frames[3]["frame"] == [4, 0, 0]
