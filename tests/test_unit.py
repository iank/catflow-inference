from catflow_inference.main import centerxywh_to_xyxy


def test_box_translation():
    # This test validates the bounding box translation
    box = [100.20, 83.83, 53.0, 99]

    start_x, start_y, end_x, end_y = centerxywh_to_xyxy(box)
    assert start_x == 73
    assert start_y == 34
    assert end_x == 153
    assert end_y == 182
