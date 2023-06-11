from __future__ import annotations
from copyreg import pickle
from dataclasses import dataclass
import abc
import pickle
import sys
from typing import Optional

import pygame as pg
import numpy as np

from psoann.ann import MultiLayerPerceptron, MultiLayerPerceptronWeights

from neurofish.typing import Position, random_position
from neurofish.aquarium import Aquarium
from neurofish.chips import Chips
from neurofish.fish import FishEnvironment, Neurofish


class FishSprite(pg.sprite.Sprite):
    def __init__(self, neurofish: Neurofish):
        super(FishSprite, self).__init__()
        self.neurofish = neurofish

        self.image = pg.image.load("./assets/fish.png")
        self.orig_image = self.image

        # position = scale_position(10, self.neurofish.position)
        position = self.neurofish.position
        self.rect = self.image.get_rect(center=position)

    def update(self):
        self.image = pg.transform.rotozoom(
            self.orig_image, -self.neurofish.angle_deg(), 1
        )
        # self.rect.center = scale_position(10, self.neurofish.position)
        self.rect.center = self.neurofish.position


class ChipsSprite(pg.sprite.Sprite):
    def __init__(self, chips: Chips):
        super(ChipsSprite, self).__init__()
        self.chips = chips

        self.image = pg.image.load("./assets/chips.png")

        # position = scale_position(10, self.chips.position)
        position = self.chips.position
        self.rect = self.image.get_rect(center=position)


class Event(abc.ABC):
    pass


class Listener(abc.ABC):
    def handle(self, event: Event) -> None:
        pass


class EventManager:
    def __init__(self):
        self.listeners = []

    def add_listener(self, listener) -> None:
        self.listeners.append(listener)

    def remove_listener(self, listener) -> None:
        self.listeners.remove(listener)

    def publish(self, event) -> None:
        for listener in self.listeners:
            listener.handle(event)


@dataclass
class TickEvent(Event):
    pass


@dataclass
class QuitEvent(Event):
    pass


@dataclass
class AddChipsEvent(Event):
    chips: Chips


@dataclass
class RemoveChipsEvent(Event):
    chips: Chips


class RunnerListener(Listener):
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager

        self.clock = pg.time.Clock()
        self.is_running = True

    def run(self):
        while self.is_running:
            self.event_manager.publish(TickEvent())
            self.clock.tick(30)
        pg.quit()

    def handle(self, event: Event) -> None:
        if isinstance(event, QuitEvent):
            self.is_running = False


class AquariumListener(Listener):
    def __init__(self, event_manager: EventManager, aquarium: Aquarium):
        self.event_manager = event_manager
        self.aquarium = aquarium

    def handle(self, event: Event) -> None:
        if isinstance(event, TickEvent):
            for fish in self.aquarium.fishes:
                fish.update(self.aquarium._get_fish_env(fish.position))
                # eat chips if possible
                chips = self.aquarium._get_chips(fish.position)
                for c in chips:
                    event = RemoveChipsEvent(c)
                    self.event_manager.publish(event)
                    fish.num_chips_eaten += 1
        elif isinstance(event, AddChipsEvent):
            self.aquarium.add_chips(event.chips)
        elif isinstance(event, RemoveChipsEvent):
            self.aquarium.remove_chips(event.chips)


class KeyboardMouseListener(Listener):
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager

    def handle(self, event: Event) -> None:
        if isinstance(event, TickEvent):
            # handle all keyboard or mouse events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.event_manager.publish(QuitEvent())
                elif event.type == pg.MOUSEBUTTONUP:
                    self.event_manager.publish(
                        AddChipsEvent(chips=Chips(pg.mouse.get_pos()))
                    )


class ViewListener(Listener):
    def __init__(
        self,
        event_manager: EventManager,
        neurofishes: list[Neurofish],
        chips: list[Chips],
        width: int,
        height: int,
    ):
        self.event_manager = event_manager

        self.screen = pg.display.set_mode([width, height])
        pg.init()

        self.fishes_group = pg.sprite.Group([FishSprite(fish) for fish in neurofishes])
        self.chips_group = pg.sprite.Group([ChipsSprite(c) for c in chips])

    def handle(self, event: Event) -> None:
        if isinstance(event, TickEvent):
            # draw background
            self.screen.fill((255, 255, 255))
            # draw the chips
            self.chips_group.update()
            self.chips_group.draw(self.screen)
            # draw the fishes
            self.fishes_group.update()
            self.fishes_group.draw(self.screen)
            # update the full display
            pg.display.flip()
        elif isinstance(event, AddChipsEvent):
            self.chips_group.add(ChipsSprite(event.chips))
        elif isinstance(event, RemoveChipsEvent):
            for sprite in self.chips_group.sprites():
                if sprite.chips == event.chips:
                    sprite.kill()


def main(path):
    screen_width = 960
    screen_height = 540

    width = screen_width
    height = screen_height

    ann_weights = pickle.load(open(path, "rb"))

    neurofishes = [
        Neurofish(
            ann_weights,
            (screen_width / 2, screen_height / 2),
            vision_resolution=ann_weights.shape[0] - 2,
        )
    ]
    num_chips = 5
    chips = [Chips.random(width, height) for _ in range(num_chips)]

    aquarium = Aquarium(screen_width, screen_height, neurofishes, chips=chips)

    # Runtime
    event_manager = EventManager()

    runner = RunnerListener(event_manager)
    view = ViewListener(event_manager, neurofishes, chips, screen_width, screen_height)
    keyboard_listener = KeyboardMouseListener(event_manager)
    aquarium_listener = AquariumListener(event_manager, aquarium)

    event_manager.add_listener(runner)
    event_manager.add_listener(keyboard_listener)
    event_manager.add_listener(aquarium_listener)
    event_manager.add_listener(view)

    runner.run()


def main2(path):
    width = 960
    height = 540

    ann_weights = pickle.load(open(path, "rb"))

    fish = Neurofish(
        ann_weights, (100, 100), vision_resolution=ann_weights.shape[0] - 2
    )
    num_chips = 10

    chips = [Chips.random(width, height) for _ in range(num_chips)]
    aquarium = Aquarium(width, height, [fish], chips)
    for i in range(10000):
        aquarium.update()
        # print(fish.position)
        # print(fish.angle_deg())
        print(i, fish.num_chips_eaten)


if __name__ == "__main__":
    path = sys.argv[1]
    main(path)
