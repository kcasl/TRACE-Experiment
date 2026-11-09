from trace.relation import CLOSE, EpisodeMemory, OPEN, PICK, PUT, TOGGLE_OFF, TOGGLE_ON, infer_relation


def _memory(*records) -> EpisodeMemory:
    memory = EpisodeMemory()
    for action, cls, success, t, role, receptacle in records:
        memory.record(action, cls, success, t, role=role, receptacle_cls=receptacle)
    return memory


def test_open_then_put_or_close_is_same():
    memory = _memory((OPEN, "Drawer", True, 1, "object", None))
    put = infer_relation(PUT, "Drawer", memory)
    close = infer_relation(CLOSE, "Drawer", memory)
    other = infer_relation(CLOSE, "Cabinet", memory)
    assert (put.mode, put.rule) == ("SAME", "R1")
    assert (close.mode, close.rule) == ("SAME", "R1")
    assert other.mode == "NONE"
    assert memory.entries[0].role == "receptacle"


def test_put_then_close_is_same_and_processing_pickup_is_same():
    placed = _memory((PUT, "Drawer", True, 2, "receptacle", None))
    assert infer_relation(CLOSE, "Drawer", placed).rule == "R2"

    heated = _memory((PUT, "Apple", True, 3, "object", "Microwave"))
    assert infer_relation(PICK, "Apple", heated).rule == "R4"
    assert infer_relation(PICK, "Apple", heated).mode == "SAME"


def test_put_on_table_then_pickup_is_different():
    memory = _memory((PUT, "Apple", True, 4, "object", "DiningTable"))
    relation = infer_relation(PICK, "Apple", memory)
    assert (relation.mode, relation.rule) == ("DIFFERENT", "R5")


def test_second_pickup_of_pick_two_is_different():
    memory = _memory((PICK, "Apple", True, 1, "object", None))
    relation = infer_relation(PICK, "Apple", memory, pick_two_second=True)
    assert (relation.mode, relation.rule) == ("DIFFERENT", "R5")
    assert infer_relation(PICK, "Apple", memory).mode == "NONE"


def test_toggle_and_failed_actions():
    lamps = _memory((TOGGLE_ON, "DeskLamp", True, 1, "object", None))
    assert infer_relation(TOGGLE_OFF, "DeskLamp", lamps).rule == "R3"
    failed = _memory((OPEN, "Drawer", False, 1, "receptacle", None))
    assert infer_relation(CLOSE, "Drawer", failed).mode == "NONE"
    assert failed.entries == []
