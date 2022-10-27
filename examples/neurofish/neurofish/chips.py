from dataclasses import dataclass
import numpy as np

from .typing import Position


@dataclass
class Chips:
    position: Position
    radius: int = 200

    def can_be_eaten(self, position: Position) -> bool:
        return np.norm(np.mat(self.position) - np.mat(position)) <= self.radius
