import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites

# Class to represent Pacman character
class Pacman(Entity):
    # Initializing the Pacman object
    def __init__(self, node):
        Entity.__init__(self, node )
        self.name = PACMAN    
        self.color = YELLOW
        # Initial movement direction
        self.direction = LEFT
        # Position between starting node and the node to the left
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)

    # Resetting to the starting state
    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    # Setting state when pacman dies
    def die(self):
        self.alive = False
        self.direction = STOP

    # Updating state every frame
    def update(self, dt):
        # Updating current animation frame	
        self.sprites.update(dt)
        # Moving based on current direction and speed
        self.position += self.directions[self.direction]*self.speed*dt
        # Getting desired direction from keyboard input
        direction = self.getValidKey()
        # Checking if reached or passed the center of target
        if self.overshotTarget():
            self.node = self.target
            # Checking if the node is a portal
            # Moving to the node linked by the portal
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            # Determining the next target based on input direction
            self.target = self.getNewTarget(direction)
            # If the desired direction is valid
            # Update movement direction
            if self.target is not self.node:
                self.direction = direction
            # If the desired direction is invalid
            # Try to continue moving in the current direction or stop
            else:
                self.target = self.getNewTarget(self.direction)
            if self.target is self.node:
                self.direction = STOP
            # Setting exact position to the center of the current node
            self.setPosition()
        # If has not reached the target yet
        # Check if the key opposite to the current direction
        # Allow to reverse direction
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()

    # Getting the desired direction from keyboard arrow keys
    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP  

    # Checking for collision with any pellet in a list of pellets
    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None    
    
    # Checking for collision with a single ghost object
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    # Performing a circular collision check between Pacman and specified object
    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False
