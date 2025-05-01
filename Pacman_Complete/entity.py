import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from random import randint

# Class for different objects representation
class Entity(object):
    # Object initialization
    def __init__(self, node):
        # Object type (pacman, ghost)
        self.name = None
        # Mapping directions to vectors
        self.directions = {UP:Vector2(0, -1),DOWN:Vector2(0, 1), 
                          LEFT:Vector2(-1, 0), RIGHT:Vector2(1, 0), STOP:Vector2()}
        # Current direction
        self.direction = STOP
        # Movement speed
        self.setSpeed(100)
        # Radius for drawing object
        self.radius = 10
        # Radius for collision checks
        self.collideRadius = 5
        # Default color
        self.color = WHITE
        # Display flag
        self.visible = True
        # Use of portals
        self.disablePortal = False
        # Target position
        self.goal = None
        # Function to choose next direction
        self.directionMethod = self.randomDirection
        # Setting initial node and position
        self.setStartNode(node)
        # Object image
        self.image = None

    # Matching entity's position to it's current node position
    def setPosition(self):
        self.position = self.node.position.copy()

    # Updating entity's position and direction over time interval
    # bases on current direction and speed
    def update(self, dt):
        self.position += self.directions[self.direction]*self.speed*dt
        # If reached or passed target position
        # get valid direction and choose next direction based on specified method
        if self.overshotTarget():
            self.node = self.target
            directions = self.validDirections()
            direction = self.directionMethod(directions)
            # Portal travel handling
            if not self.disablePortal:
                if self.node.neighbors[PORTAL] is not None:
                    self.node = self.node.neighbors[PORTAL]
            # Setting new target based on chosen direction
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            # Updating current position
            self.setPosition()
    
    # Check if it possible to move in specified direction from the current node
    def validDirection(self, direction):
        if direction is not STOP:
            if self.name in self.node.access[direction]:
                if self.node.neighbors[direction] is not None:
                    return True
        return False

    # Getting new node by moving in specified direction (neighboring or current)
    def getNewTarget(self, direction):
        if self.validDirection(direction):
            return self.node.neighbors[direction]
        return self.node

    # Checking if entity reached or passed the target
    def overshotTarget(self):
        if self.target is not None:
            vec1 = self.target.position - self.node.position
            vec2 = self.position - self.node.position
            node2Target = vec1.magnitudeSquared()
            node2Self = vec2.magnitudeSquared()
            return node2Self >= node2Target
        return False

    # Reversing entity direction and swap node/target
    def reverseDirection(self):
        self.direction *= -1
        temp = self.node
        self.node = self.target
        self.target = temp
    
    # Checking if specified direction is the opposite of current direction
    def oppositeDirection(self, direction):
        if direction is not STOP:
            if direction == self.direction * -1:
                return True
        return False

    # Getting a list of valid directions
    def validDirections(self):
        directions = []
        for key in [UP, DOWN, LEFT, RIGHT]:
            if self.validDirection(key):
                if key != self.direction * -1:
                    directions.append(key)
        if len(directions) == 0:
            directions.append(self.direction * -1)
        return directions

    # Choosing random direction from the list
    def randomDirection(self, directions):
        return directions[randint(0, len(directions)-1)]

    # Choosing direction that leads closest to the target position
    def goalDirection(self, directions):
        distances = []
        for direction in directions:
            vec = self.node.position  + self.directions[direction]*TILEWIDTH - self.goal
            distances.append(vec.magnitudeSquared())
        index = distances.index(min(distances))
        return directions[index]

    # Setting initial node
    def setStartNode(self, node):
        self.node = node
        self.startNode = node
        self.target = node
        self.setPosition()

    # Placing entity exactly between current node and its neighbor
    # in specified direction
    def setBetweenNodes(self, direction):
        if self.node.neighbors[direction] is not None:
            self.target = self.node.neighbors[direction]
            self.position = (self.node.position + self.target.position) / 2.0

    # Resetting to the initial state
    def reset(self):
        self.setStartNode(self.startNode)
        self.direction = STOP
        self.speed = 100
        self.visible = True

    # Setting speed based on tile size to maintain consistency
    def setSpeed(self, speed):
        self.speed = speed * TILEWIDTH / 16

    # Displaying entity on the game screen
    def render(self, screen):
        if self.visible:
            # If image is defined, adjust based on the dimensions
            if self.image is not None:
                adjust = Vector2(TILEWIDTH, TILEHEIGHT) / 2
                p = self.position - adjust
                screen.blit(self.image, p.asTuple())
            # If image is not defined, use default circle
            else:
                p = self.position.asInt()
                pygame.draw.circle(screen, self.color, p, self.radius)