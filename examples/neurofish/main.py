from dataclasses import dataclass
import pygame as pg
from numpy.linalg import norm
from numpy import mat

Position = tuple([float, float])


@dataclass
class Neurofish:
    position: Position
    angle: float = 0

    def update(self):
        self.angle += 5
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


def main():
    screen_width = 960
    screen_height = 540

    pg.init()
    screen = pg.display.set_mode([screen_width, screen_height])

    clock = pg.time.Clock()

    neurofishes = [Neurofish((screen_width / 2, screen_height / 2))]
    aquarium = Aquarium(neurofishes)

    fishes_group = pg.sprite.Group([FishSprite(fish) for fish in neurofishes])
    chips_group = pg.sprite.Group()

    is_running = True
    while is_running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                is_running = False
            if event.type == pg.MOUSEBUTTONUP:
                pos = pg.mouse.get_pos()
                # add chips
                aquarium.add_chips(pos)
                chips_group.add(ChipsSprite(pos))

        # update all the aquarium
        aquarium.update()

        # background
        screen.fill((255, 255, 255))
        # draw the chips
        chips_group.update()
        chips_group.draw(screen)
        # draw the fishes
        fishes_group.update()
        fishes_group.draw(screen)
        # update the full display
        pg.display.flip()

        clock.tick(30)

    pg.quit()


if __name__ == "__main__":
    main()
