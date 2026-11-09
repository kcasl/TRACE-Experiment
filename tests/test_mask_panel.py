import numpy as np

from experiments.thor_rollout import instance_mask, render_mask_panel


class FakeEvent:
    def __init__(self, *, mask=None, box=None, seg=None, color=None, size=40):
        self.frame = np.full((size, size, 3), 80, dtype=np.uint8)
        self.instance_masks = {}
        self.instance_detections2D = {}
        self.instance_segmentation_frame = seg
        self.object_id_to_color = {}
        if mask is not None:
            self.instance_masks["Sink|a"] = mask
        if box is not None:
            self.instance_detections2D["Sink|a"] = box
        if color is not None:
            self.object_id_to_color["Sink|a"] = color


def test_instance_mask_uses_pixels_not_the_bbox():
    mask = np.tri(40, 40, dtype=bool)
    event = FakeEvent(mask=mask, box=[0, 0, 40, 40])
    got = instance_mask(event, "Sink|a")
    assert got is not None
    assert bool(got[39, 0]) and not bool(got[0, 39])


def test_instance_mask_from_segmentation_frame():
    seg = np.zeros((20, 20, 3), dtype=np.uint8)
    seg[5:12, 5:12] = (10, 20, 30)
    event = FakeEvent(size=20, seg=seg, color=(10, 20, 30))
    event.frame = np.full((20, 20, 3), 80, dtype=np.uint8)
    got = instance_mask(event, "Sink|a")
    assert got is not None
    assert int(got.sum()) == 49


def test_panel_fills_mask_pixels_only():
    mask = np.tri(40, 40, dtype=bool)
    event = FakeEvent(mask=mask, box=[0, 0, 40, 40])
    panel = render_mask_panel(event, ["Sink|a"], "Sink|a", crop="none")
    assert panel is not None
    body = panel[32:]
    assert int(body[39, 0, 0]) > 120
    assert int(body[0, 39, 0]) < 100


def test_panel_does_not_fill_bbox_when_mask_missing():
    event = FakeEvent(box=[5, 5, 25, 25])
    panel = render_mask_panel(event, ["Sink|a"], "Sink|a", crop="none")
    body = panel[32:]
    assert abs(int(body[15, 15, 0]) - 80) < 8
