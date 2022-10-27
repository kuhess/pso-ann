from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from psoann import ann

from .typing import Position
from .chips import Chips


def is_in_sector(
    point: Position,
    origin: Position,
    sector_angles_rad: tuple([float, float]),
    sector_radius: float,
):
    if sector_angles_rad[1] - sector_angles_rad[0] >= 2 * np.pi:
        return True

    x = point[0] - origin[0]
    y = point[1] - origin[1]
    angle_rad = np.arctan2(y, x)
    distance = np.sqrt(x * x + y * y)

    if distance > sector_radius:
        return False

    return sector_angles_rad[0] <= angle_rad and angle_rad <= sector_angles_rad[1]


@dataclass
class FishEnvironment:
    width: int
    height: int
    chips: list[Chips]


@dataclass
class Neurofish:
    ann: ann.MultiLayerPerceptronWeights
    position: Position
    angle_deg: float = 0

    max_speed: float = 10
    max_rotation: float = 10

    vision_angle_rad: float = np.pi / 2
    vision_distance: float = 50

    def angle_rad(self):
        return np.pi * self.angle_deg / 180

    @classmethod
    def create_random(cls, ann_shape: list[int], position: Position) -> Neurofish:
        return cls(
            ann=ann.MultiLayerPerceptronWeights.create_random(ann_shape),
            position=position,
        )

    def _get_num_vision_sector(self):
        return self.ann.num_inputs()

    def get_vision_data(self, env: FishEnvironment):
        semi_vision_angle_rad = self.vision_angle_rad / 2

        min_angle_rad = self.angle_rad() - semi_vision_angle_rad
        max_angle_rad = self.angle_rad() + semi_vision_angle_rad

        num_sectors = self._get_num_vision_sector()
        sector_angle_rad = (max_angle_rad - min_angle_rad) / num_sectors

        data = np.zeros(num_sectors)
        for i in range(num_sectors):
            min_sector_angle_rad = min_angle_rad + i * sector_angle_rad
            max_sector_angle_rad = min_angle_rad + (i + 1) * sector_angle_rad

            for c in env.chips:
                if is_in_sector(
                    c.position,
                    self.position,
                    (min_sector_angle_rad, max_sector_angle_rad),
                    self.vision_distance,
                ):
                    data[i] = 1
                    break

        return data
