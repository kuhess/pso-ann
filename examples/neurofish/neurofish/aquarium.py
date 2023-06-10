from __future__ import annotations
from dataclasses import dataclass, field

from neurofish.chips import Chips
from neurofish.fish import FishEnvironment, Neurofish
from neurofish.typing import Position


@dataclass
class Aquarium:
    width: int
    height: int
    fishes: list[Neurofish] = field(default_factory=list)
    chips: list[Chips] = field(default_factory=list)

    def add_chips(self, chips: Chips) -> Aquarium:
        self.chips.append(chips)
        return self

    def remove_chips(self, chips) -> Aquarium:
        self.chips.remove(chips)
        return self

    def _get_chips(self, position: Position) -> list[Chips]:
        eaten = []
        for c in self.chips:
            if c.can_be_eaten(position):
                eaten.append(c)
        return eaten

    def _get_fish_env(self, position: Position):
        return FishEnvironment(self.width, self.height, self.chips)

    def update(self):
        for fish in self.fishes:
            fish.update(self._get_fish_env(fish.position))
            # eat chips if possible
            chips = self._get_chips(fish.position)
            for c in chips:
                self.remove_chips(c)
                fish.num_chips_eaten += 1
