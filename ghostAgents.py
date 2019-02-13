# ghostAgents.py
# --------------
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


from directions import Directions
from actions import Actions
import random
from util import manhattanDistance
import util
import numpy as np


class GhostAgent:

    def __init__(self, player, args):
        self.player = player

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()

        dir = self.player.dir
        reverse = Actions.reverseDirection(dir)

        legalActions=[ 
            x for x in Actions.getPossibleActions(self.player.pos, state.layout.walls)
            if x != reverse and x != Directions.STOP
        ]

        for a in legalActions:
            dist[a] = 1.0
        dist.normalize()
        return dist


def minItem(list, prediction) :
    
    minValue = 9999999
    minItem = None
    for item in list :
        v = prediction(item)
        if minValue > v :
            minValue = v
            minItem = item
    return minItem



class AstarGhost(GhostAgent) :


    def getAction(self, state) :


        pacmans = state.alivePacmans()


        def astar(s, t, walls, width, height) :
            openList = [s]
            mapShape = (width, height)
            close_table = np.zeros(mapShape)
            h_table = np.zeros(mapShape)
            g_table = np.zeros(mapShape)

            parent = {} 


            
            while(len(openList) > 0 and not close_table[t[0], t[1]]) :
                minF = minItem(openList, lambda x : g_table[x[0] , x[1]] + h_table[x[0] , x[1]])
                openList.remove(minF)
                # 当前f值最小的点
                fx, fy = minF
                close_table[fx, fy] = 1

                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if (abs(i) != abs(j)) :
                            x, y = fx + i, fy + j

                            if not (x >= 0 and y >= 0 and x < width and y < height):
                                continue

                            if walls[x][y] :
                                continue
                            
                            if close_table[x,y] :
                                continue
                            
                            if not (x, y) in openList :
                                openList.append((x, y))
                                g_table[x, y] = g_table[fx, fy] + 1
                                h_table[x, y] =  manhattanDistance((x, y), t) 
                                parent[(x, y)] = (fx, fy) 
                            else :
                                if g_table[x,y] > g_table[fx, fy] + 1 :
                                    parent[(x, y)] = (fx, fy) 
                                    g_table[x, y] = g_table[fx, fy] + 1
            path = []
            m = t
            while m in parent.keys():
                p = parent[m]
                path.append(p)
                m = p
            path.reverse()
            return path
        
        paths = [astar(self.player.pos, p.pos, state.layout.walls, state.layout.width, state.layout.height) for p in  pacmans]

        # print(paths)
        # import sys
        # sys.exit(0)

        minLenPath = None
        minLen = 999

        for p in paths:
            if minLen > len(p) :
                minLen = len(p)
                minLenPath = p 

        if len(p) == 1 or len(p) == 0:
            return Directions.STOP

        tx, ty = p[1]
        sx, sy = self.player.pos 
        action = Actions.vectorToDirection((tx - sx, ty - sy) )
        return action
               



class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, player, args):
        self.player = player 
        self.prob_attack = 0.8 
        self.prob_scaredFlee = 0.8 

    def getDistribution(self, state):
        # Read variables from state
        # ghostState = state.getGhostState(self.index)
        # legalActions = state.getLegalActions(self.index)
        player = self.player
        # legalActions=Actions.getPossibleActions(self.player.pos, state.layout.walls)

        reverse = Actions.reverseDirection(dir)
        legalActions=[ 
            x for x in Actions.getPossibleActions(self.player.pos, state.layout.walls)
            if x != reverse and x != Directions.STOP
        ]

        actionVectors = [Actions.directionToVector(
            a, 1) for a in legalActions]
        newPositions = [(self.player.pos[0] + a[0], self.player.pos[1] + a[1]) for a in actionVectors]


        pacmanPositions = state.getPacmanPositions()

        def distance(pos):
            return manhattanDistance(pos, self.player.pos)

        pacmanPositions.sort(key = distance)
        pacmanPosition = pacmanPositions[0]
           
        # Select best actions given the state
        distancesToPacman = [manhattanDistance(
            pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
            legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist
