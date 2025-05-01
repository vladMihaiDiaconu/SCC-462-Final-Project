import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from modes import ModeController
from sprites import GhostSprites

# Class to represent ghost instance
class Ghost(Entity):
    # Initializing parent and extra settings
    def __init__(self, node, pacman=None, blinky=None):
        Entity.__init__(self, node)
        # Entity identifier
        self.name = GHOST
        # Poins for eating ghost in freight mode
        self.points = 200
        # Target position
        self.goal = Vector2()
        # Goal-seeking movement method
        self.directionMethod = self.goalDirection
        # Pacman entity
        self.pacman = pacman
        # Controlling ghost's current behavior
        self.mode = ModeController(self)
        # Referencing blinky object
        self.blinky = blinky
        # Starting node
        self.homeNode = node

    # Resetting to the initial state
    def reset(self):
        Entity.reset(self)
        self.points = 200
        self.directionMethod = self.goalDirection

    # Updating ghost state each frame (appearance, mode controller, target, movement)
    def update(self, dt):
        self.sprites.update(dt)
        self.mode.update(dt)
        if self.mode.current is SCATTER:
            self.scatter()
        elif self.mode.current is CHASE:
            self.chase()
        Entity.update(self, dt)

    # Default scatter behavior
    def scatter(self):
        self.goal = Vector2()

    # Default chase behavior
    def chase(self):
        self.goal = self.pacman.position

    # Setting goal to the spawn position
    def spawn(self):
        self.goal = self.spawnNode.position

    # Setting node to return to when eaten
    def setSpawnNode(self, node):
        self.spawnNode = node

    # Initiating SPAWN mode (when eaten)
    def startSpawn(self):
        self.mode.setSpawnMode()
        if self.mode.current == SPAWN:
            # Increasing speed when returning
            self.setSpeed(150)
            self.directionMethod = self.goalDirection
            self.spawn()

    # Initiating FREIGHT mode (when Pacman eats a power pellet)
    def startFreight(self):
        self.mode.setFreightMode()
        if self.mode.current == FREIGHT:
            # Reducing speed during freight mode
            self.setSpeed(50)
            self.directionMethod = self.randomDirection         

    # Returning ghost to normal speed and movement after Freight/Spawn ends
    def normalMode(self):
        self.setSpeed(100)
        self.directionMethod = self.goalDirection
        self.homeNode.denyAccess(DOWN, self)



# Class to represent Blinky ghost
class Blinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = BLINKY
        self.color = RED
        self.sprites = GhostSprites(self)

# Class to represent Pinky ghost
class Pinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = PINKY
        self.color = PINK
        self.sprites = GhostSprites(self)

    # Setting scatter target (top-right corner)
    def scatter(self):
        self.goal = Vector2(TILEWIDTH*NCOLS, 0)

    # Setting chase behavior based on pacman current direction
    def chase(self):
        # Calculate target 4 tiles ahead of Pacman's position in current direction
        self.goal = self.pacman.position + self.pacman.directions[self.pacman.direction] * TILEWIDTH * 4

# Class to represent Inky ghost
class Inky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = INKY
        self.color = TEAL
        self.sprites = GhostSprites(self)

    # Setting scatter target (bottom-right corner)
    def scatter(self):
        self.goal = Vector2(TILEWIDTH*NCOLS, TILEHEIGHT*NROWS)

    # Setting chase behavior based on pacman current direction
    def chase(self):
        # Calculate a point 2 tiles ahead of Pacman
        vec1 = self.pacman.position + self.pacman.directions[self.pacman.direction] * TILEWIDTH * 2
        # Calculate vector from Blinky to that point, then double it
        vec2 = (vec1 - self.blinky.position) * 2
        # Inky's target is Blinky's position plus the doubled vector
        self.goal = self.blinky.position + vec2

# Class to represent Clyde ghost
class Clyde(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = CLYDE
        self.color = ORANGE
        self.sprites = GhostSprites(self)

    # Setting scatter target (bottom-left corner)
    def scatter(self):
        self.goal = Vector2(0, TILEHEIGHT*NROWS)

    # Setting chase behavior based on pacman current direction
    def chase(self):
        # Calculate vector from Clyde to Pacman
        d = self.pacman.position - self.position
        # Calculate squared distance
        ds = d.magnitudeSquared()
        # If Pacman is within 8 tiles distance (squared)
        # revert to scatter behavior
        if ds <= (TILEWIDTH * 8)**2:
            self.scatter()
        # If Pacman is far away
        # target 4 tiles ahead of Pacman 
        else:
            self.goal = self.pacman.position + self.pacman.directions[self.pacman.direction] * TILEWIDTH * 4

# Class to manage the group of all four ghosts
class GhostGroup(object):
    # Initializing the group
    def __init__(self, node, pacman):
        self.blinky = Blinky(node, pacman)
        self.pinky = Pinky(node, pacman)
        self.inky = Inky(node, pacman, self.blinky)
        self.clyde = Clyde(node, pacman)
        self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]

    # Iiterating directly over the GhostGroup object 
    def __iter__(self):
        return iter(self.ghosts)

    # Updating all ghosts in the group
    def update(self, dt):
        for ghost in self:
            ghost.update(dt)

    # Putting all ghosts into Freight mode
    def startFreight(self):
        for ghost in self:
            ghost.startFreight()
        self.resetPoints()

    # Setting the spawn node for all ghosts
    def setSpawnNode(self, node):
        for ghost in self:
            ghost.setSpawnNode(node)

    # Double the points for eating consecutive ghosts in Freight mode
    def updatePoints(self):
        for ghost in self:
            ghost.points *= 2

    # Resetting the point value back to the initial value
    def resetPoints(self):
        for ghost in self:
            ghost.points = 200

    # Making all ghosts invisible
    def hide(self):
        for ghost in self:
            ghost.visible = False

    # Making all ghosts visible
    def show(self):
        for ghost in self:
            ghost.visible = True

    # Resetting all ghosts to their starting positions and states
    def reset(self):
        for ghost in self:
            ghost.reset()

    # Rendering all ghosts to the screen
    def render(self, screen):
        for ghost in self:
            ghost.render(screen)

