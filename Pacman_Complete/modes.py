from constants import *

# Class to manage the main cycle between Scatter and Chase modes
class MainMode(object):
    def __init__(self):
        self.timer = 0
        self.scatter()

    # Updating the main mode timer and switch modes if necessary
    def update(self, dt):
        self.timer += dt
        if self.timer >= self.time:
            if self.mode is SCATTER:
                self.chase()
            elif self.mode is CHASE:
                self.scatter()

    # Activating Scatter mode
    def scatter(self):
        self.mode = SCATTER
        # Setting the duration (seconds) for Scatter mode
        self.time = 7
        self.timer = 0

    # Activating Chase mode
    def chase(self):
        self.mode = CHASE
        # Setting the duration (seconds) for Chase mode
        self.time = 20
        self.timer = 0

# Class to control the specific behavior mode
class ModeController(object):
    # Initializing the controller for a specific entity
    def __init__(self, entity):
        self.timer = 0
        self.time = None
        self.mainmode = MainMode()
        self.current = self.mainmode.mode
        self.entity = entity 

    # Updating the entity's current mode based on timers and game state
    def update(self, dt):
        self.mainmode.update(dt)
        if self.current is FREIGHT:
            self.timer += dt
            if self.timer >= self.time:
                self.time = None
                self.entity.normalMode()
                self.current = self.mainmode.mode
        elif self.current in [SCATTER, CHASE]:
            self.current = self.mainmode.mode

        if self.current is SPAWN:
            if self.entity.node == self.entity.spawnNode:
                self.entity.normalMode()
                self.current = self.mainmode.mode

    # Switching the entity to Freight mode (when Pacman eats a power pellet)
    def setFreightMode(self):
        if self.current in [SCATTER, CHASE]:
            self.timer = 0
            self.time = 7
            self.current = FREIGHT
        elif self.current is FREIGHT:
            self.timer = 0

    # Switching the entity to Spawn mode (when ghost is eaten during Freight)
    def setSpawnMode(self):
        if self.current is FREIGHT:
            self.current = SPAWN