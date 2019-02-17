# layout.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from grid import Grid
import os
import random
from functools import reduce

VISIBILITY_MATRIX_CACHE = {}


class Layout:
    """
    A Layout manages the static information about the game board.
    """

    def __init__(self, layoutText, pacmanFeast=dict()):
        self.width = len(layoutText[0])
        self.height = len(layoutText)
        self.walls = Grid(self.width, self.height, False)
        self.food = Grid(self.width, self.height, False)
        self.capsules = []
        # pacmans' and ghosts' positions
        self.agentPositions = []
        # number of ghosts
        self.numGhosts = 0
        self.numPacman = 0
        self.processLayoutText(layoutText, pacmanFeast)
        self.layoutText = layoutText
        # number of foods
        self.totalFood = len(self.food.asList())
        self.pacmanFeast = pacmanFeast
        # self.initializeVisibilityMatrix()

    def getNumGhosts(self):
        return self.numGhosts

    def initializeVisibilityMatrix(self):
        global VISIBILITY_MATRIX_CACHE
        if reduce(str.__add__, self.layoutText) not in VISIBILITY_MATRIX_CACHE:
            from game import Directions
            vecs = [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]
            dirs = [Directions.NORTH, Directions.SOUTH,
                    Directions.WEST, Directions.EAST]
            vis = Grid(self.width, self.height, {Directions.NORTH: set(), Directions.SOUTH: set(
            ), Directions.EAST: set(), Directions.WEST: set(), Directions.STOP: set()})
            for x in range(self.width):
                for y in range(self.height):
                    if self.walls[x][y] == False:
                        for vec, direction in zip(vecs, dirs):
                            dx, dy = vec
                            nextx, nexty = x + dx, y + dy
                            while (nextx + nexty) != int(nextx) + int(nexty) or not self.walls[int(nextx)][int(nexty)]:
                                vis[x][y][direction].add((nextx, nexty))
                                nextx, nexty = x + dx, y + dy
            self.visibility = vis
            VISIBILITY_MATRIX_CACHE[reduce(str.__add__, self.layoutText)] = vis
        else:
            self.visibility = VISIBILITY_MATRIX_CACHE[
                reduce(str.__add__, self.layoutText)]

    def isWall(self, pos):
        x, col = pos
        return self.walls[x][col]

    def getRandomLegalPosition(self):
        x = random.choice(list(range(self.width)))
        y = random.choice(list(range(self.height)))
        while self.isWall((x, y)):
            x = random.choice(list(range(self.width)))
            y = random.choice(list(range(self.height)))
        return (x, y)

    def getRandomCorner(self):
        poses = [(1, 1), (1, self.height - 2), (self.width - 2, 1),
                 (self.width - 2, self.height - 2)]
        return random.choice(poses)

    def getFurthestCorner(self, pacPos):
        poses = [(1, 1), (1, self.height - 2), (self.width - 2, 1),
                 (self.width - 2, self.height - 2)]
        dist, pos = max([(manhattanDistance(p, pacPos), p) for p in poses])
        return pos

    def isVisibleFrom(self, ghostPos, pacPos, pacDirection):
        row, col = [int(x) for x in pacPos]
        return ghostPos in self.visibility[row][col][pacDirection]

    def __str__(self):
        return "\n".join(self.layoutText)

    def deepCopy(self):
        return Layout(self.layoutText[:], self.pacmanFeast)

    def processLayoutText(self, layoutText, pacmanFeast=dict()):
        """
        Coordinates are flipped from the input format to the (x,y) convention here

        The shape of the maze.  Each character
        represents a different type of object.
         % - Wall
         . - Food
         o - Capsule
         G - Ghost
         P - Pacman
        Other characters are ignored.
        """
        maxY = self.height - 1
        super = 0
        pacmanCount = 0
        for y in range(self.height):
            for x in range(self.width):
                layoutChar = layoutText[maxY - y][x]
                if str(layoutChar) == '4':
                    if str(pacmanCount) in pacmanFeast:
                        if pacmanFeast[str(pacmanCount)] == True:
                            super = 1
                    pacmanCount += 1
                self.processLayoutChar(x, y, layoutChar, super)
                super = 0
        # self.agentPositions.sort()
        self.agentPositions = [(i == 0, pos, super)
                               for i, pos, super in self.agentPositions]

    def processLayoutChar(self, x, y, layoutChar, super=0):
        layoutChar = str(layoutChar)
        # wall
        if layoutChar == '%':
            self.walls[x][y] = True
        # bean
        elif layoutChar == '.':
            self.food[x][y] = True
        # capsules
        elif layoutChar == 'o':
            self.capsules.append((x, y))
        # pacman
        elif layoutChar == 'P':
            self.agentPositions.append((0, (x, y), super))
            self.numPacman += 1
        # ghost
        elif layoutChar in ['G']:
            self.agentPositions.append((1, (x, y), super))
            self.numGhosts += 1
        elif layoutChar in ['1', '2', '3', '4']:
            self.agentPositions.append((int(layoutChar), (x, y)))
            self.numGhosts += 1

    def refreshLayout(self, dots):
        layoutText = self.layoutText.copy()
        maxY = self.height - 1
        for y in range(len(layoutText[0])):
            for x in range(len(layoutText)):
                layoutChar = str(layoutText[x][y])
                if layoutChar in ['2', '3', '4', '5']:
                    layoutText[x][y] = 0
        pacDots = dots['pacDots']
        powerPellets = dots['powerPellets']
        pacman = dots['pacman']
        ghosts = dots['ghosts']
        pacmanFeast = dots['pacmanFeast']
        for i in range(len(pacDots)):
            layoutText[pacDots[i]['x']][pacDots[i]['y']] = 2
        for i in range(len(powerPellets)):
            layoutText[powerPellets[i]['x']][powerPellets[i]['y']] = 3
        # for i in range(len(pacman)):
        for value in pacman.values():
            layoutText[value['x']][value['y']] = 4
        for value in ghosts.values():
            layoutText[value['x']][value['y']] = 5
        return Layout(layoutText, pacmanFeast)


def getLayout(name, back=2):
    if name.endswith('.lay'):
        layout = tryToLoad('layouts/' + name)
        if layout == None:
            layout = tryToLoad(name)
    else:
        layout = tryToLoad('layouts/' + name + '.lay')
        if layout == None:
            layout = tryToLoad(name + '.lay')
    if layout == None and back >= 0:
        curdir = os.path.abspath('.')
        os.chdir('..')
        layout = getLayout(name, back - 1)
        os.chdir(curdir)
    return layout

# 初始化地图


def initLayout(m):
    return Layout(m)


def tryToLoad(fullname):
    if(not os.path.exists(fullname)):
        return None
    f = open(fullname)
    try:
        # print('m: ', [line.strip() for line in f])
        return Layout([line.strip() for line in f])
    finally:
        f.close()
