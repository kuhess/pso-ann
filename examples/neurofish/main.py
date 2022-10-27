from __future__ import annotations
from dataclasses import dataclass
import abc

import pygame as pg
import numpy as np
from numpy.linalg import norm
from numpy import mat, pi, zeros, sin, cos, reshape
from numpy.random import uniform

from psoann import ann


def random_position(size):
    x = uniform(0, size[0])
    y = uniform(0, size[1])
    return (x, y)


def cart2bary(corner1, corner2, corner3, point):
    T = mat(
        [
            [corner1[0] - corner3[0], corner2[0] - corner3[0]],
            [corner1[1] - corner3[1], corner2[1] - corner3[1]],
        ]
    )
    A = mat([[point[0] - corner3[0]], [point[1] - corner3[1]]])
    L = T.I * A
    return (L[0, 0], L[1, 0], 1 - L[0, 0] - L[1, 0])


def get_bary_in(corner1, corner2, corner3, point):
    l = cart2bary(corner1, corner2, corner3, point)
    if (0 <= l[0] <= 1) and (0 <= l[1] <= 1) and (0 <= l[2] <= 1):
        return l
    else:
        return None


Position = tuple([float, float])


@dataclass
class Chips:
    position: Position
    radius: int = 200

    def can_be_eaten(self, position: Position) -> bool:
        return norm(mat(self.position) - mat(position)) <= self.radius


@dataclass
class FishEnvironment:
    width: int
    height: int
    chips: list[Chips]


@dataclass
class Neurofish:
    ann: ann.MultiLayerPerceptronWeights
    position: Position
    angle: float = 0
    num_chips: int = 0
    max_speed: float = 10
    max_rotation: float = 10
    vision_range: tuple([float, float]) = (-pi / 2, pi / 2)
    vision_distance: float = 50

    @classmethod
    def create_random(cls, ann_shape: list[int], position: Position) -> Neurofish:
        return cls(
            ann=ann.MultiLayerPerceptronWeights.create_random(ann_shape),
            position=position,
        )

    def get_inputs(self, env: FishEnvironment):
        min_angle = self.angle + self.vision_range[0]
        max_angle = self.angle + self.vision_range[1]

        num_inputs = self.ann.num_inputs()
        num_vision_angles = num_inputs
        vision_step = (max_angle - min_angle) / num_vision_angles
        inputs = zeros(num_inputs)
        for i in range(num_inputs):
            vision_angle_min = min_angle + i * vision_step
            vision_angle_max = min_angle + (i + 1) * vision_step

            p1 = self.position

            p2 = (
                p1[0] + self.vision_distance * np.cos(vision_angle_min),
                p1[1] + self.vision_distance * np.sin(vision_angle_min),
            )

            p3 = (
                p1[0] + self.vision_distance * np.cos(vision_angle_max),
                p1[1] + self.vision_distance * np.sin(vision_angle_max),
            )

            dist = None
            for chip in env.chips:
                l = get_bary_in(p1, p2, p3, chip.position)
                if l is not None:
                    tmp = self.vision_distance * l[0]
                    if not dist or tmp < dist:
                        dist = tmp
            if dist is not None:
                inputs[i] = dist

        return inputs

    def update(self, env: FishEnvironment):
        inputs = self.get_inputs(env)
        outputs = ann.MultiLayerPerceptron.run(self.ann, inputs)

        # should be in ]0;1[
        speed_ratio = outputs[0]
        print(speed_ratio)

        # should be in ]-1;1[
        rotation_ratio = (outputs[1] * 2) - 1
        rotation = self.max_rotation * rotation_ratio  # todo tidy rotation range

        self.speed = self.max_speed * speed_ratio
        self.angle = self.angle + rotation

        x = self.position[0] + self.speed * cos(self.angle)
        y = self.position[1] + self.speed * sin(self.angle)

        self.position = (x % env.width, y % env.height)


class Aquarium:
    def __init__(self, width: int, height: int, fishes: list[Neurofish]):
        self.width = width
        self.height = height
        self.fishes = fishes
        self.chips = []

    def add_chips(self, chips: Chips):
        self.chips.append(chips)

    def remove_chips(self, chips: Chips):
        self.chips.remove(chips)

    def update(self):
        for fish in self.fishes:
            env = FishEnvironment(self.width, self.height, self.chips)
            fish.update(env)
            # woot
            to_remove = []
            for chips in self.chips:
                if chips.can_be_eaten(fish.position):
                    to_remove.append(chips)
            [self.remove_chips(c) for c in to_remove]


class FishSprite(pg.sprite.Sprite):
    def __init__(self, neurofish: Neurofish):
        super(FishSprite, self).__init__()
        self.neurofish = neurofish

        self.pos = neurofish.position
        self.angle = neurofish.angle

        self.orig_image = pg.image.load("./assets/fish.png")

        self.image = self.orig_image
        self.rect = self.image.get_rect(center=self.pos)

    def update(self):
        self.pos = self.neurofish.position
        self.angle = self.neurofish.angle

        self.image = pg.transform.rotozoom(self.orig_image, self.angle, 1)
        self.rect = self.image.get_rect(center=self.pos)


class ChipsSprite(pg.sprite.Sprite):
    def __init__(self, chips: Chips):
        super(ChipsSprite, self).__init__()
        self.chips = chips

        self.image = pg.image.load("./assets/chips.png")
        self.rect = self.image.get_rect(center=self.chips.position)


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
    def __init__(
        self, event_manager: EventManager, width, height, neurofishes: list[Neurofish]
    ):
        self.event_manager = event_manager
        self.width = width
        self.height = height
        self.neurofishes = neurofishes
        self.chips = []

    def handle(self, event: Event) -> None:
        if isinstance(event, TickEvent):
            for fish in self.neurofishes:
                env = FishEnvironment(self.width, self.height, self.chips)
                fish.update(env)
        if isinstance(event, AddChipsEvent):
            self.chips.append(event.chips)
        elif isinstance(event, RemoveChipsEvent):
            self.chips.remove(event.chips)


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
        width: int,
        height: int,
    ):
        self.event_manager = event_manager

        self.screen = pg.display.set_mode([width, height])
        pg.init()

        self.fishes_group = pg.sprite.Group([FishSprite(fish) for fish in neurofishes])
        self.chips_group = pg.sprite.Group()

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
            self.chips_group.remove(ChipsSprite(event.chips))


def main():
    screen_width = 960
    screen_height = 540

    ann_shape = [4, 3, 2]

    neurofishes = [
        Neurofish.create_random(ann_shape, (screen_width / 2, screen_height / 2))
    ]

    # Runtime
    event_manager = EventManager()

    runner = RunnerListener(event_manager)
    view = ViewListener(event_manager, neurofishes, screen_width, screen_height)
    keyboard_listener = KeyboardMouseListener(event_manager)
    aquarium_listener = AquariumListener(
        event_manager, screen_width, screen_height, neurofishes
    )

    event_manager.add_listener(runner)
    event_manager.add_listener(keyboard_listener)
    event_manager.add_listener(aquarium_listener)
    event_manager.add_listener(view)

    runner.run()


if __name__ == "__main__":
    main()
