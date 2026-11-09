import numpy as np

from experiments.scenes import far_fridge, open_drawers_to_close, same_room_drawers
from trace.consequence.critic import ConsequenceCritic
from trace.consequence.planner import CLOSE, PUT


def _pair(scene, action):
    world, template, ids = scene()
    costs = ConsequenceCritic().cost_matrix(world, action, ids, template)
    return costs.C


def test_same_room_put_regret_is_two_steps():
    costs = _pair(same_room_drawers, PUT)
    assert costs.shape == (2, 3)
    np.testing.assert_allclose(costs[:, 0], [0.0, 2.0])
    np.testing.assert_allclose(costs[:, 1], [2.0, 0.0])
    assert np.all(costs >= -1e-9)


def test_closing_the_wrong_drawer_is_cheaper_than_a_wrong_put():
    costs = _pair(open_drawers_to_close, CLOSE)
    np.testing.assert_allclose(costs[:, 0], [0.0, 1.0])


def test_far_fridge_regret_is_not_symmetric():
    # Column 0: the reference fridge is in the kitchen.
    # Putting in the garage fridge costs a round trip, 22 above the local put.
    # Column 1: the reference fridge is already in the garage, so the trip is
    # paid by the correct action too. The local mistake adds only put + pickup.
    costs = _pair(far_fridge, PUT)
    np.testing.assert_allclose(costs[:, 0], [0.0, 22.0])
    np.testing.assert_allclose(costs[:, 1], [2.0, 0.0])


def test_both_fridges_far_makes_every_mistake_a_round_trip():
    from experiments.scenes import split_fridges

    costs = _pair(split_fridges, PUT)
    np.testing.assert_allclose(costs[:, 0], [0.0, 12.0])
    np.testing.assert_allclose(costs[:, 1], [12.0, 0.0])
