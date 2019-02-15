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
from util import manhattanDistance, astar

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






class AstarGhost(GhostAgent) :
    def getRandom(self, state):
        dist = util.Counter()

        dir = self.player.dir
        reverse = Actions.reverseDirection(dir)

        legalActions=[ 
            x for x in Actions.getPossibleActions(self.player.pos, state.layout.walls)
            if x != reverse and x != Directions.STOP
        ]
        if len(legalActions) == 0:
            return Directions.STOP
        return random.choice(legalActions)

    def getAction(self, state) :


        pacmans = state.alivePacmans()



        
        # paths = [astar(self.player.pos, p.pos, state.layout.walls, state.layout.width, state.layout.height) for p in  pacmans]

        nearestPacman = None
        nearestPath = None
        minPath = 1e10 

        for pacman in pacmans : 
            path = astar(self.player.pos, pacman.pos, state.layout.walls, state.layout.width, state.layout.height)
            if minPath > len(path) :
                nearestPacman = pacman
                nearestPath = path
                minPath = len(path)
        
        rnd=True



        if len(nearestPath) > 0 :
            if len(nearestPath) > 3:
                return self.getRandom(state)
            tx, ty = nearestPath[0]
            sx, sy = self.player.pos 
            action = Actions.vectorToDirection((tx - sx, ty - sy) )

            reverseAct = False
            if nearestPath[0] in state.capsules and len(nearestPath) == 2:
               reverseAct = True
          
            if nearestPacman.super:
                reverseAct = True
            
            if reverseAct :
                nbrs = Actions.getLegalNeighbors(self.player.pos, state.layout.walls)

                maxDist = -1
                bestChoice = None
                for n in nbrs :
                    dist = util.distance( nearestPacman.pos, n)
                    if dist > maxDist :
                        maxDist = dist
                        bestChoice = n

                bestAction = Actions.vectorToDirection((bestChoice[0] - sx, bestChoice[1]-sy))
                action = bestAction
                # print(state)
                # print(nearestPath)
                # print(bestAction)
                # sys.exit()
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
