from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from psoann.ann import MultiLayerPerceptron, MultiLayerPerceptronWeights

from .typing import Position
from .chips import Chips


def mod_rad(angle_rad: float) -> float:
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


@dataclass
class FishEnvironment:
    width: int
    height: int
    chips: list[Chips]


@dataclass
class FishThought:
    speed_ratio: float
    rotation_ratio: float


@dataclass
class Neurofish:
    ann_weights: MultiLayerPerceptronWeights
    position: Position
    angle_rad: float = 0

    speed: float = 0
    max_speed: float = 3

    rotation_rad: float = 0
    max_rotation_rad: float = np.pi / 8

    vision_resolution: int = 5
    vision_angle_rad: float = np.pi / 2
    vision_distance: float = 500  # np.Inf

    num_chips_eaten: int = 0
    distance: float = 0

    def angle_deg(self):
        return np.rad2deg(self.angle_rad)

    @classmethod
    def create_random(cls, ann_shape: list[int], position: Position) -> Neurofish:
        return cls(
            ann_weights=MultiLayerPerceptronWeights.create_random(ann_shape),
            position=position,
        )

    def get_vision_data(self, env: FishEnvironment):
        num_sectors = self.vision_resolution
        if len(env.chips) == 0:
            return np.zeros((num_sectors,))

        semi_vision_angle_rad = self.vision_angle_rad / 2
        min_angle_rad = -semi_vision_angle_rad
        max_angle_rad = semi_vision_angle_rad

        chips_positions = np.asarray([c.position for c in env.chips])
        fish_position = np.asarray(self.position)

        chips_rel_positions = chips_positions - fish_position

        angles = mod_rad(
            np.arctan2(
                chips_rel_positions[:, 1],
                chips_rel_positions[:, 0],
            )
            - self.angle_rad
        )

        in_front_mask = np.where((angles >= min_angle_rad) & (angles <= max_angle_rad))

        distances = np.linalg.norm(chips_rel_positions[in_front_mask], axis=1)
        ok_distances_mask = np.where(distances < self.vision_distance)

        sector_angles = np.linspace(min_angle_rad, max_angle_rad, num_sectors + 1)

        sectors = np.digitize(
            (angles[in_front_mask])[ok_distances_mask], bins=sector_angles
        )

        _ndx = np.argsort(sectors)
        unique_sectors, _pos = np.unique(sectors[_ndx], return_index=True)
        minimum_distances = np.minimum.reduceat(
            distances[ok_distances_mask][_ndx], _pos
        )

        sector_distances = np.zeros((num_sectors,))
        for (idx, d) in zip(unique_sectors, minimum_distances):
            sector_distances[idx - 1] = 1 - (d / self.vision_distance)

        return sector_distances

    def update(self, env: FishEnvironment) -> Neurofish:
        vision = self.get_vision_data(env)
        inputs = np.hstack([vision, self.speed, self.rotation_rad])
        outputs = MultiLayerPerceptron.run(self.ann_weights, inputs)

        speed_ratio = outputs[0]
        rotation_ratio = outputs[1]

        new_speed = self.max_speed * speed_ratio
        new_rotation_rad = self.max_rotation_rad * (2 * rotation_ratio - 1)

        new_angle_rad = mod_rad(self.angle_rad + new_rotation_rad)
        new_position = (
            (self.position[0] + new_speed * np.cos(new_angle_rad)) % env.width,
            (self.position[1] + new_speed * np.sin(new_angle_rad)) % env.height,
        )

        self.angle_rad = new_angle_rad
        self.position = new_position

        self.speed = new_speed
        self.rotation_rad = new_rotation_rad

        self.distance += new_speed
