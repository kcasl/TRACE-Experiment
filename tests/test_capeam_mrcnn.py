from experiments.capeam_mrcnn import iou


def test_iou_identical_boxes():
    box = [10, 10, 30, 30]
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
