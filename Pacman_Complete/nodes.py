import pygame
from vector import Vector2
from constants import *
import numpy as np

# Class to represent a single point in the maze grid
class Node(object):
    # Initializing a Node object
    def __init__(self, x, y):
        # Position of the node 
        self.position = Vector2(x, y)
        # References to neighboring nodes in each direction 
        self.neighbors = {UP:None, DOWN:None, LEFT:None, RIGHT:None, PORTAL:None}
        # Defining which entity types are allowed to move *from* this node *to* a neighbor in a given direction
        self.access = {UP:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT], 
                       DOWN:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT], 
                       LEFT:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT], 
                       RIGHT:[PACMAN, BLINKY, PINKY, INKY, CLYDE, FRUIT]}

    # Removing permission to move from this node in a specified direction
    def denyAccess(self, direction, entity):
        if entity.name in self.access[direction]:
            self.access[direction].remove(entity.name)

    # Granting permission to move from this node in a specified direction
    def allowAccess(self, direction, entity):
        if entity.name not in self.access[direction]:
            self.access[direction].append(entity.name)

    # Displaying the node and its connections
    def render(self, screen):
        for n in self.neighbors.keys():
            if self.neighbors[n] is not None:
                line_start = self.position.asTuple()
                line_end = self.neighbors[n].position.asTuple()
                pygame.draw.line(screen, WHITE, line_start, line_end, 4)
                pygame.draw.circle(screen, RED, self.position.asInt(), 12)

# Class to manage the collection of all nodes
class NodeGroup(object):
    def __init__(self, level):
        self.level = level
        # Nodes objects and its coordinates
        self.nodesLUT = {}
        # Symbols in the maze file that represent nodes
        self.nodeSymbols = ['+', 'P', 'n']
        # Symbols in the maze file that represent paths
        self.pathSymbols = ['.', '-', '|', 'p']
        # Reading Maze layout from the specified file
        data = self.readMazeFile(level)
        # Creating Node objects based on the maze data
        self.createNodeTable(data)
        # Connecting nodes horizontally based on the maze data
        self.connectHorizontally(data)
        # Connecting nodes vertically based on the maze data
        self.connectVertically(data)
        # Main ghost home node
        self.homekey = None

    # Loading maze data from a text file
    def readMazeFile(self, textfile):
        return np.loadtxt(textfile, dtype='<U1')

    # Creating Node objects for each node symbol found in the maze data
    def createNodeTable(self, data, xoffset=0, yoffset=0):
        for row in list(range(data.shape[0])):
            for col in list(range(data.shape[1])):
                if data[row][col] in self.nodeSymbols:
                    x, y = self.constructKey(col+xoffset, row+yoffset)
                    self.nodesLUT[(x, y)] = Node(x, y)

    # Converting tile coordinates (column, row) to pixel coordinates (x, y)
    def constructKey(self, x, y):
        return x * TILEWIDTH, y * TILEHEIGHT

    # Connecting nodes horizontally based on path symbols between them
    def connectHorizontally(self, data, xoffset=0, yoffset=0):
        for row in list(range(data.shape[0])):
            key = None
            for col in list(range(data.shape[1])):
                if data[row][col] in self.nodeSymbols:
                    if key is None:
                        key = self.constructKey(col+xoffset, row+yoffset)
                    else:
                        otherkey = self.constructKey(col+xoffset, row+yoffset)
                        self.nodesLUT[key].neighbors[RIGHT] = self.nodesLUT[otherkey]
                        self.nodesLUT[otherkey].neighbors[LEFT] = self.nodesLUT[key]
                        key = otherkey
                elif data[row][col] not in self.pathSymbols:
                    key = None

    # Connecting nodes vertically based on path symbols between them
    def connectVertically(self, data, xoffset=0, yoffset=0):
        dataT = data.transpose()
        for col in list(range(dataT.shape[0])):
            key = None
            for row in list(range(dataT.shape[1])):
                if dataT[col][row] in self.nodeSymbols:
                    if key is None:
                        key = self.constructKey(col+xoffset, row+yoffset)
                    else:
                        otherkey = self.constructKey(col+xoffset, row+yoffset)
                        self.nodesLUT[key].neighbors[DOWN] = self.nodesLUT[otherkey]
                        self.nodesLUT[otherkey].neighbors[UP] = self.nodesLUT[key]
                        key = otherkey
                elif dataT[col][row] not in self.pathSymbols:
                    key = None

    # Getting starting node
    def getStartTempNode(self):
        nodes = list(self.nodesLUT.values())
        return nodes[0]

    # Establishing a portal connection between two nodes
    def setPortalPair(self, pair1, pair2):
        key1 = self.constructKey(*pair1)
        key2 = self.constructKey(*pair2)
        if key1 in self.nodesLUT.keys() and key2 in self.nodesLUT.keys():
            self.nodesLUT[key1].neighbors[PORTAL] = self.nodesLUT[key2]
            self.nodesLUT[key2].neighbors[PORTAL] = self.nodesLUT[key1]

    # Creating the nodes for the ghost home area based on a predefined layout
    def createHomeNodes(self, xoffset, yoffset):
        homedata = np.array([['X','X','+','X','X'],
                             ['X','X','.','X','X'],
                             ['+','X','.','X','+'],
                             ['+','.','+','.','+'],
                             ['+','X','X','X','+']])

        self.createNodeTable(homedata, xoffset, yoffset)
        self.connectHorizontally(homedata, xoffset, yoffset)
        self.connectVertically(homedata, xoffset, yoffset)
        self.homekey = self.constructKey(xoffset+2, yoffset)
        return self.homekey

    # Connecting a node within the ghost home area to a node in the main maze grid
    def connectHomeNodes(self, homekey, otherkey, direction):     
        key = self.constructKey(*otherkey)
        self.nodesLUT[homekey].neighbors[direction] = self.nodesLUT[key]
        self.nodesLUT[key].neighbors[direction*-1] = self.nodesLUT[homekey]

    # Retrieving a Node object using its exact pixel coordinates
    def getNodeFromPixels(self, xpixel, ypixel):
        if (xpixel, ypixel) in self.nodesLUT.keys():
            return self.nodesLUT[(xpixel, ypixel)]
        return None

    # Retrieving a Node object using tile grid coordinates (column, row)
    def getNodeFromTiles(self, col, row):
        x, y = self.constructKey(col, row)
        if (x, y) in self.nodesLUT.keys():
            return self.nodesLUT[(x, y)]
        return None

    # Denying movement access for a specific entity at a specific tile location and direction
    def denyAccess(self, col, row, direction, entity):
        node = self.getNodeFromTiles(col, row)
        if node is not None:
            node.denyAccess(direction, entity)
    
    # Allowing movement access for a specific entity at a specific tile location and direction
    def allowAccess(self, col, row, direction, entity):
        node = self.getNodeFromTiles(col, row)
        if node is not None:
            node.allowAccess(direction, entity)

    # Denying movement access for a list of entities at a specific tile location and direction
    def denyAccessList(self, col, row, direction, entities):
        for entity in entities:
            self.denyAccess(col, row, direction, entity)

    # Allowing movement access for a list of entities at a specific tile location and direction
    def allowAccessList(self, col, row, direction, entities):
        for entity in entities:
            self.allowAccess(col, row, direction, entity)

    # Specifically denying downward movement access for an entity at the main ghost home node
    def denyHomeAccess(self, entity):
        self.nodesLUT[self.homekey].denyAccess(DOWN, entity)

    # Specifically allowing downward movement access for an entity at the main ghost home node
    def allowHomeAccess(self, entity):
        self.nodesLUT[self.homekey].allowAccess(DOWN, entity)

    # Denying downward home access for a list of entities
    def denyHomeAccessList(self, entities):
        for entity in entities:
            self.denyHomeAccess(entity)

    # Allowing downward home access for a list of entities
    def allowHomeAccessList(self, entities):
        for entity in entities:
            self.allowHomeAccess(entity)

    # Rendering all nodes in the group
    def render(self, screen):
        for node in self.nodesLUT.values():
            node.render(screen)