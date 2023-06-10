import pytest
import numpy as np

from neurofish.fish import FishEnvironment, Neurofish
from neurofish.chips import Chips

NUM_SECTORS = 5

# def test_create_random():
#     fish = Neurofish.create_random(ann_shape=[3, 2, 1], position=(18, -5))
#     assert fish is not None


# @pytest.mark.parametrize(
#     "pos,origin,sector_angles,sector_radius,expected",
#     [
#         ((5, 0), (0, 0), (-np.pi / 4, np.pi / 4), 5, True),
#         ((5, 0), (0, 0), (-np.pi / 4, np.pi / 4), 0, False),
#         ((0, 0), (0, 0), (-np.pi / 4, np.pi / 4), 1, True),
#         ((-1, 0), (0, 0), (-np.pi / 4, np.pi / 4), 10, False),
#         ((-5, 0), (0, 0), (-np.pi, np.pi), 10, True),
#         ((-5, 0), (0, 0), (-3 * np.pi / 4, 3 * np.pi / 4), 10, False),
#         ((11, 6), (10, 5), (0, np.pi / 2), 2, True),
#     ],
# )
# def test_is_in_sector(pos, origin, sector_angles, sector_radius, expected):
#     res = view_sector(
#         pos,
#         origin,
#         sector_angles,
#         sector_radius,
#     )
#     assert res == expected


# def test_get_vision_data():
#     num_sectors = 5
#     fish = Neurofish.create_random(ann_shape=[num_sectors, 1], position=(0, 0))
#     fish.angle_rad = 0
#     fish.vision_angle_rad = np.pi / 2 + 0.01
#     fish.vision_distance = 10

#     env = FishEnvironment(
#         1000,
#         1000,
#         [
#             Chips((5, 0)),
#             Chips((2, 2)),
#             Chips((2, -2)),
#             Chips((1, 1)),
#             Chips((-5, 0)),
#             Chips((1, 0)),
#         ],
#     )
#     res = fish.get_vision_data(env)

#     np.testing.assert_array_equal(res, np.asarray([1, 0, 1, 0, 1]))


def fish(position, angle_rad, num_sectors):
    neurofish = Neurofish.create_random(
        ann_shape=[num_sectors, 1],
        position=position,
    )
    neurofish.angle_rad = angle_rad
    neurofish.vision_resolution = num_sectors
    neurofish.vision_angle_rad = np.pi / 2 + 0.01
    neurofish.vision_distance = 10
    return neurofish


def env(chips):
    return FishEnvironment(width=1000, height=1000, chips=chips)


@pytest.mark.parametrize(
    "fish, env, vision",
    [
        (fish((0, 0), np.deg2rad(0), 1), env([]), [0]),
        (
            fish((0, 0), np.deg2rad(0), 1),
            env([Chips((1.6, 0))]),
            np.asarray([10 - 1.6]) / 10,
        ),
        (
            fish((0, 0), np.deg2rad(0), 3),
            env([Chips((1.6, 0))]),
            np.asarray([0, 10 - 1.6, 0]) / 10,
        ),
        (
            fish((0, 0), np.arctan2(4, 3), 3),
            env([Chips((3, 4))]),
            np.asarray([0, 10 - 5, 0]) / 10,
        ),
        (
            fish((100, -200), np.deg2rad(-90), 1),
            env([Chips((100, -202))]),
            np.asarray([10 - 2]) / 10,
        ),
        (
            fish((0, 0), np.deg2rad(-180), 1),
            env([Chips((-1.2, 0))]),
            np.asarray([10 - 1.2]) / 10,
        ),
        (
            fish((0, 0), np.deg2rad(-180), 3),
            env([Chips((-1.2, 0))]),
            np.asarray([0, 10 - 1.2, 0]) / 10,
        ),
    ],
)
def test_get_vision_data2(fish, env, vision):
    res = fish.get_vision_data(env)
    np.testing.assert_array_almost_equal(res, np.asarray(vision))
