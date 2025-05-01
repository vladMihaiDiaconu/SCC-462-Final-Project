import pygame
from entity import Entity
from constants import *
from sprites import FruitSprites

# Class to represent fruit instance
class Fruit(Entity):
    # Initializing parent and extra settings
    def __init__(self, node, level=0):
        Entity.__init__(self, node)
        # Entity identifier
        self.name = FRUIT
        # Default color
        self.color = GREEN
        # Duration the fruit remains on screen
        self.lifespan = 5
        # Interval time tracker
        self.timer = 0
        # Flag to remove fruit
        self.destroy = False
        # Calculating points based on level
        self.points = 100 + level*20
        # Placing fruit in the middle of the path to the RIGHT of its spawn node
        self.setBetweenNodes(RIGHT)
        # Initializing sprite for this fruit
        self.sprites = FruitSprites(self, level)

    # Updating fruit's state based on time interval
    def update(self, dt):
        self.timer += dt
        # Checking if timer has exceeded the fruit's lifespan
        if self.timer >= self.lifespan:
            # Flag for destruction
            self.destroy = True