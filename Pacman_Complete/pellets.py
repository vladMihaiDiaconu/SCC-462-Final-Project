import pygame
from vector import Vector2
from constants import *
import numpy as np

# Class to represent a standard pellet
class Pellet(object):
    # Initializing a Pellet object
    def __init__(self, row, column):
        self.name = PELLET
        self.position = Vector2(column*TILEWIDTH, row*TILEHEIGHT)
        self.color = WHITE
        self.radius = int(2 * TILEWIDTH / 16)
        self.collideRadius = 2 * TILEWIDTH / 16
        # Points for eating this pellet
        self.points = 10
        self.visible = True
    
    # Displaying the pellet on the screen if it is visible
    def render(self, screen):
        if self.visible:
            adjust = Vector2(TILEWIDTH, TILEHEIGHT) / 2
            p = self.position + adjust
            pygame.draw.circle(screen, self.color, p.asInt(), self.radius)


# Class to represents a power pellet that makes ghosts vulnerable
class PowerPellet(Pellet):
    # Initializing a PowerPellet object
    def __init__(self, row, column):
        Pellet.__init__(self, row, column)
        self.name = POWERPELLET
        self.radius = int(8 * TILEWIDTH / 16)
        # Points for eating power pellet
        self.points = 50
        # Flashing effect
        self.flashTime = 0.2
        self.timer= 0
    
    # Updating the power pellet's state each frame 
    def update(self, dt):
        self.timer += dt
        # Check if the timer has reached the flash interval duration
        # Makes it flash on and off
        if self.timer >= self.flashTime:
            self.visible = not self.visible
            self.timer = 0

# Class to manage the collection of all pellets
class PelletGroup(object):
    # Initializing the PelletGroup
    def __init__(self, pelletfile):
        self.pelletList = []
        self.powerpellets = []
        self.createPelletList(pelletfile)
        self.numEaten = 0

    # Updating the state of pellets in the group
    def update(self, dt):
        for powerpellet in self.powerpellets:
            powerpellet.update(dt)
    
    # Reading the layout file and creates Pellet/PowerPellet objects accordingly
    def createPelletList(self, pelletfile):
        data = self.readPelletfile(pelletfile)        
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in ['.', '+']:
                    self.pelletList.append(Pellet(row, col))
                elif data[row][col] in ['P', 'p']:
                    pp = PowerPellet(row, col)
                    self.pelletList.append(pp)
                    self.powerpellets.append(pp)

    # Reading the specified maze/pellet layout      
    def readPelletfile(self, textfile):
        return np.loadtxt(textfile, dtype='<U1')
    
    # Checking if all pellets in the group have been eaten
    def isEmpty(self):
        if len(self.pelletList) == 0:
            return True
        return False
    
    # Rendering all pellets in the group
    def render(self, screen):
        for pellet in self.pelletList:
            pellet.render(screen)