from constants import *

# Class for frame-based animations
class Animator(object):
    # Initialization of instance variables
    def __init__(self, frames=[], speed=20, loop=True):
        # List of animation frames
        self.frames = frames
        # Index of current frame
        self.current_frame = 0
        # Playback speed
        self.speed = speed
        # Repeating animation after finishing
        self.loop = loop
        # Delta time to control frame switching
        self.dt = 0
        # Non-looping animantion completion flag
        self.finished = False

    # Resetting animation to its initial state
    def reset(self):
        # Going back to the first frame
        self.current_frame = 0
        # Flag as not finished
        self.finished = False

    # Updating animation based on delta time
    def update(self, dt):
        # If animation has not finished
        # calculate if it is time for the next frame
        if not self.finished:
            self.nextFrame(dt)
        # If it is the last frame
        # and looping is enabled
        # go back to the beginning
        if self.current_frame == len(self.frames):
            if self.loop:
                self.current_frame = 0
            # if looping is disabled
            # flag as finished and stay on the last frame
            else:
                self.finished = True
                self.current_frame -= 1
        # Returning current frame's data
        return self.frames[self.current_frame]

    # Controling when to advance to the next frame
    def nextFrame(self, dt):
        # Adding interval time since the last update
        self.dt += dt
        # Checking if enough time has passed based on the specified speed
        # Moving to the next frame
        # Resetting time interval
        if self.dt >= (1.0 / self.speed):
            self.current_frame += 1
            self.dt = 0                  
