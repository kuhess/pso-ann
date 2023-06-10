from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .typing import Position, random_position


@dataclass
class Chips:
    position: Position
    radius: float = 15

    def can_be_eaten(self, position: Position) -> bool:
        return np.linalg.norm(np.mat(self.position) - position) <= self.radius

    @classmethod
    def random(
        cls, width, height, exclude_pos: Position = None, exclude_radius: int = 20
    ) -> Chips:
        if exclude_pos is None:
            pos = random_position(width, height)
        else:
            is_ok = False
            while not is_ok:
                pos = random_position(width, height)
                dist = np.linalg.norm(np.asarray(pos) - np.asarray(exclude_pos))
                if dist > exclude_radius:
                    is_ok = True
        return cls(position=pos)
