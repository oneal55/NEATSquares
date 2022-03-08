import pygame
from pygame.locals import *
import sys
import random
import neat
import os
import math
import time

pygame.init()
vec = pygame.math.Vector2  # 2 for two dimensional
height = 400
width = 700
acc = 0.5
friction = -0.12
fps = 30

FramePerSec = pygame.time.Clock()


def distance(posn1, posn2):
    return math.sqrt((posn1[0] - posn2[0])**2 + (posn1[1] - posn2[1])**2)


class Platform(pygame.sprite.Sprite):
    def __init__(self, size_tup, pos):
        super().__init__()
        self.surf = pygame.Surface(size_tup)
        self.surf.fill((0, 255, 0))
        self.rect = self.surf.get_rect(center=(pos[0], pos[1]))
        self.point = True

    def move(self):
        pass


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # self.image = pygame.image.load("character.png")
        self.surf = pygame.Surface((30, 30))
        self.surf.fill(((random.randint(50, 200)), (random.randint(50, 200)), (random.randint(50, 200))))
        self.rect = self.surf.get_rect(center=(40, width - 40))

        self.jumping = False
        self.score = 0
        self.travelled = 0
        self.recentScore = False
        self.hasStarted = False
        self.direction = "None"
        self.count = 0

        self.pos = vec((40, height - 50))
        self.vel = vec(0, 0)
        self.acc = vec(0, 0.5)

    def move_right(self):
        self.acc = vec(0, 0.5)
        self.acc.x = acc

        self.acc.x += self.vel.x * friction
        self.vel += self.acc
        self.travelled += self.vel.x
        self.pos += self.vel

        if self.pos.x > width / 2 - 15:
            self.pos.x = width / 2 - 15

        self.rect.midbottom = self.pos

    def move_left(self):
        self.acc = vec(0, 0.5)
        self.acc.x = -acc

        self.acc.x += self.vel.x * friction
        self.vel += self.acc
        self.travelled += self.vel.x
        self.pos += self.vel

        if self.pos.x >= width / 2 - 15:
            self.pos.x = width / 2 - 15

        self.rect.midbottom = self.pos

    def jump(self, platforms):
        if not self.jumping:
            self.jumping = True
            self.vel.y = -12

    def update(self, platforms):
        self.acc = vec(0, 0.5)
        self.count += 1
        hits = pygame.sprite.spritecollide(self, platforms, False)
        if self.vel.y >= 0:
            if len(hits) > 0:
                if hits[0].rect.bottom > self.pos.y >= hits[0].rect.top:
                    self.vel.y = 0
                    self.pos.y = hits[0].rect.top
                    self.jumping = False
                    self.recentScore = True
                    if hits[0].point:
                        hits[0].point = False
                        self.score += 1
                        self.count = 0
                        self.recentScore = True

    def closest_platform(self, platforms):
        old = Platform((1, 1), (-5, -10))
        nearest = old
        for platform in platforms:
            if distance((self.rect.x, self.rect.y), (platform.rect.x, platform.rect.y)) <\
                    distance((self.rect.x, self.rect.y), (nearest.rect.x, nearest.rect.y))\
                    and platform.rect.x > self.rect.x:
                nearest = platform
        old.kill()
        return nearest


def eval_genomes(genomes, config):
    display_surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Game")

    platforms = pygame.sprite.Group()

    platform1 = Platform([width, 20], (width / 2, height - 10))
    platform1.point = False
    players = []
    ge = []
    nets = []
    counts = [-1] * 30

    for genome_id, genome in genomes:
        players.append(Player())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    platforms.add(platform1)

    posX = width / 2
    posY = 7 * height / 10
    for x in range(10):
        posY = posY - random.randint(-70, 70)
        while posY > 5 * height / 6 or posY < height / 4:
            posY = posY - random.randint(-70, 70)

        posX += random.randint(int(width / 8), int(width / 3))
        pl = Platform((random.randint(100, 150), 20), (posX, posY))
        platforms.add(pl)

    def check(platform, groupies):
        if pygame.sprite.spritecollideany(platform, groupies):
            return True
        else:
            for entity in groupies:
                if entity == platform:
                    continue
                if (abs(platform.rect.right - entity.rect.left) < 100)\
                        and (abs(platform.rect.left - entity.rect.right) < 100):
                    return True
            c = False

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        for i, player in enumerate(players):
            output = nets[i].activate((player.rect.y,
                                       distance((player.rect.x, player.rect.y),
                                                player.closest_platform(platforms).rect.midtop), player.vel.x))
            if output[0] > 0.5 and not player.jumping:
                player.jump(platforms)
            if output[1] > 0.5:
                player.move_right()
            if output[2] > 0.5:
                player.move_left()

            player.update(platforms)

            if player.rect.top > height:
                players.remove(player)
                ge[i].fitness += player.score * 100 + player.travelled
                ge[i].fitness -= 10
                ge.pop(i)
                nets.pop(i)
                player.kill()

            if player.rect.left < 0 and player.hasStarted:
                players.remove(player)
                ge[i].fitness += player.score * 100 + player.travelled
                ge[i].fitness -= 15
                ge.pop(i)
                nets.pop(i)
                player.kill()

        key = pygame.key.get_pressed()
        if key[K_SPACE]:
            for i, player in enumerate(players):
                players.remove(player)
                ge[i].fitness += player.score * 100 + player.travelled
                ge[i].fitness -= 50
                ge.pop(i)
                nets.pop(i)
                player.kill()

        if len(players) == 0:
            break

        while len(platforms) < 5:
            pl = Platform((random.randint(100, 150), 20), (posX, posY))
            C = True
            while C:
                posX = width + random.randint(int(width / 7), int(width / 4))
                posY = posY - random.randint(-70, 70)
                while posY > 5 * height / 6 or posY < height / 4:
                    posY = posY - random.randint(-70, 70)
                pl.rect.center = (posX, posY)
                C = check(pl, platforms)

            platforms.add(pl)

        display_surface.fill((0, 0, 0))

        front = "None"
        for player in players:
            if front == "None":
                front = player
            if player.rect.x > front.rect.x:
                front = player

        if front.rect.right >= width / 2 and front.vel.x >= 0:
            for plat in platforms:
                if front.vel.x <= 0:
                    front.vel.x = 0
                front.vel.x += front.acc.x
                plat.rect.x -= front.vel.x
                if plat.rect.right <= 0:
                    platforms.remove(plat)
                    plat.kill()
            for i, player in enumerate(players):
                if player is not front:
                    player.rect.x -= front.vel.x

        display_surface.fill((0, 0, 0))
        f = pygame.font.SysFont("Verdana", 20)
        g = f.render("Score: " + str(front.score) + "Count: " + str(len(players)), False, (123, 255, 255))
        display_surface.blit(g, (width / 2, height - 50))

        for entity in players:
            display_surface.blit(entity.surf, entity.rect)
        for entity in platforms:
            display_surface.blit(entity.surf, entity.rect)

        pygame.display.update()
        FramePerSec.tick(fps)
        for i in range(len(counts)):
            counts[i] += 1


def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path)

    pop = neat.Population(config)
    pop.run(eval_genomes, 50)

    if __name__ == '__main__':
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config.txt')
        run(config_path)

run('config')
