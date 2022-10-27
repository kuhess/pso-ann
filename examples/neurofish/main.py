from dataclasses import dataclass
import abc

import pygame as pg
from numpy.linalg import norm
from numpy import mat

Position = tuple([float, float])


@dataclass
class Neurofish:
    position: Position
    angle: float = 0

    def update(self):
        self.angle += 10
        if self.angle > 360:
            self.angle = self.angle % 360


@dataclass
class Chips:
    position: Position
    radius: int = 3

    def can_be_eaten(self, position: Position) -> bool:
        return norm(mat(self.position) - mat(position)) <= self.radius


class Aquarium:
    def __init__(self, fishes: list[Neurofish]):
        self.fishes = fishes
        self.chips = []

    def add_chips(self, position: Position):
        self.chips.append(position)

    def remove_chips(self, position: Position):
        self.chips.remove(position)

    def update(self):
        for fish in self.fishes:
            fish.update()


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
        self.rotate()

    def rotate(self):
        self.image = pg.transform.rotozoom(self.orig_image, self.angle, 1)
        self.rect = self.image.get_rect(center=self.rect.center)


class ChipsSprite(pg.sprite.Sprite):
    def __init__(self, position):
        super(ChipsSprite, self).__init__()
        self.pos = position

        self.image = pg.image.load("./assets/chips.png")
        self.rect = self.image.get_rect(center=self.pos)


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
    position: Position


@dataclass
class RemoveChipsEvent(Event):
    position: Position


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
            self.aquarium.update()
        if isinstance(event, AddChipsEvent):
            self.aquarium.add_chips(event.position)
        elif isinstance(event, RemoveChipsEvent):
            self.aquarium.remove_chips(event.position)


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
                    self.event_manager.publish(AddChipsEvent(pg.mouse.get_pos()))


class ViewListener(Listener):
    def __init__(
        self, event_manager: EventManager, aquarium: Aquarium, width: int, height: int
    ):
        self.event_manager = event_manager

        self.screen = pg.display.set_mode([width, height])
        pg.init()

        self.fishes_group = pg.sprite.Group(
            [FishSprite(fish) for fish in aquarium.fishes]
        )
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
            self.chips_group.add(ChipsSprite(event.position))
        elif isinstance(event, RemoveChipsEvent):
            self.chips_group.remove(event.position)


def main():
    screen_width = 960
    screen_height = 540

    neurofishes = [Neurofish((screen_width / 2, screen_height / 2))]
    aquarium = Aquarium(neurofishes)

    # Runtime
    event_manager = EventManager()

    runner = RunnerListener(event_manager)
    view = ViewListener(event_manager, aquarium, screen_width, screen_height)
    keyboard_listener = KeyboardMouseListener(event_manager)
    aquarium_listener = AquariumListener(event_manager, aquarium)

    event_manager.add_listener(runner)
    event_manager.add_listener(keyboard_listener)
    event_manager.add_listener(aquarium_listener)
    event_manager.add_listener(view)

    runner.run()


if __name__ == "__main__":
    main()
