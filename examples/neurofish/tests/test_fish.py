import pytest
import numpy as np

from neurofish.fish import FishEnvironment, Neurofish, is_in_sector
from neurofish.chips import Chips


# def test_create_random():
#     fish = Neurofish.create_random(ann_shape=[3, 2, 1], position=(18, -5))
#     assert fish is not None


@pytest.mark.parametrize(
    "pos,origin,sector_angles,sector_radius,expected",
    [
        ((5, 0), (0, 0), (-np.pi / 4, np.pi / 4), 5, True),
        ((5, 0), (0, 0), (-np.pi / 4, np.pi / 4), 0, False),
        ((0, 0), (0, 0), (-np.pi / 4, np.pi / 4), 1, True),
        ((-1, 0), (0, 0), (-np.pi / 4, np.pi / 4), 10, False),
        ((-5, 0), (0, 0), (-np.pi, np.pi), 10, True),
        ((-5, 0), (0, 0), (-3 * np.pi / 4, 3 * np.pi / 4), 10, False),
        ((11, 6), (10, 5), (0, np.pi / 2), 2, True),
    ],
)
def test_is_in_sector(pos, origin, sector_angles, sector_radius, expected):
    res = is_in_sector(
        pos,
        origin,
        sector_angles,
        sector_radius,
    )
    assert res == expected


def test_get_vision_data():
    fish = Neurofish.create_random(ann_shape=[5, 1], position=(0, 0))
    fish.angle = 0
    fish.vision_angle = np.pi / 2
    fish.vision_distance = 10

    env = FishEnvironment(1000, 1000, [Chips((2, 2)), Chips((5, 0)), Chips((2, -2))])
    res = fish.get_vision_data(env)

    np.testing.assert_array_equal(res, np.asarray([1, 0, 1, 0, 1]))
