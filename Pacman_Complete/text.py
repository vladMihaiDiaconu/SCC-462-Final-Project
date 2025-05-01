import pygame
from vector import Vector2
from constants import *

# Class to represent a single piece of text displayed on the screen
class Text(object):
    # Initializing the Text object
    def __init__(self, text, color, x, y, size, time=None, id=None, visible=True):
        self.id = id
        self.text = text
        self.color = color
        self.size = size
        self.visible = visible
        self.position = Vector2(x, y)
        self.timer = 0
        self.lifespan = time
        self.label = None
        self.destroy = False
        self.setupFont("PressStart2P-Regular.ttf")
        self.createLabel()

    # Loading the font file used for rendering
    def setupFont(self, fontpath):
        self.font = pygame.font.Font(fontpath, self.size)

    # Rendering the text string onto a Pygame Surface (the label)
    def createLabel(self):
        self.label = self.font.render(self.text, 1, self.color)

    # Updating the text content and re-renders the label surface
    def setText(self, newtext):
        self.text = str(newtext)
        self.createLabel()

    # Updating the timer for temporary text and checks if it should be destroyed
    def update(self, dt):
        if self.lifespan is not None:
            self.timer += dt
            if self.timer >= self.lifespan:
                self.timer = 0
                self.lifespan = None
                self.destroy = True

    # Drawing the text label onto the screen if it is visible
    def render(self, screen):
        if self.visible:
            x, y = self.position.asTuple()
            screen.blit(self.label, (x, y))


# Class to manage a collection of Text objects
class TextGroup(object):
    # Initializing the TextGroup
    def __init__(self):
        self.nextid = 10
        self.alltext = {}
        self.setupText()
        self.showText(READYTXT)

    # Adding a new Text object to the group
    def addText(self, text, color, x, y, size, time=None, id=None):
        self.nextid += 1
        self.alltext[self.nextid] = Text(text, color, x, y, size, time=time, id=id)
        return self.nextid

    # Removing a Text object from the group
    def removeText(self, id):
        self.alltext.pop(id)
    
    # Creating the standard text elements
    def setupText(self):
        size = TILEHEIGHT
        self.alltext[SCORETXT] = Text("0".zfill(8), WHITE, 0, TILEHEIGHT, size)
        self.alltext[LEVELTXT] = Text(str(1).zfill(3), WHITE, 23*TILEWIDTH, TILEHEIGHT, size)
        self.alltext[READYTXT] = Text("READY!", YELLOW, 11.25*TILEWIDTH, 20*TILEHEIGHT, size, visible=False)
        self.alltext[PAUSETXT] = Text("PAUSED!", YELLOW, 10.625*TILEWIDTH, 20*TILEHEIGHT, size, visible=False)
        self.alltext[GAMEOVERTXT] = Text("GAMEOVER!", YELLOW, 10*TILEWIDTH, 20*TILEHEIGHT, size, visible=False)
        self.addText("SCORE", WHITE, 0, 0, size)
        self.addText("LEVEL", WHITE, 23*TILEWIDTH, 0, size)

    # Updating all Text objects in the group
    def update(self, dt):
        for tkey in list(self.alltext.keys()):
            self.alltext[tkey].update(dt)
            if self.alltext[tkey].destroy:
                self.removeText(tkey)

    # Making a specific text message visible, hiding others
    def showText(self, id):
        self.hideText()
        self.alltext[id].visible = True

    # Hiding the standard messages
    def hideText(self):
        self.alltext[READYTXT].visible = False
        self.alltext[PAUSETXT].visible = False
        self.alltext[GAMEOVERTXT].visible = False

    # Updating the score display text
    def updateScore(self, score):
        self.updateText(SCORETXT, str(score).zfill(8))

    # Updating the level display text
    def updateLevel(self, level):
        self.updateText(LEVELTXT, str(level + 1).zfill(3))

    # Updating the text content of a specific Text object
    def updateText(self, id, value):
        if id in self.alltext.keys():
            self.alltext[id].setText(value)

    # Rendering all visible Text objects in the group
    def render(self, screen):
        for tkey in list(self.alltext.keys()):
            self.alltext[tkey].render(screen)
