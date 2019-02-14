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
from util import manhattanDistance, astar
import util
import numpy as np
import sys
import ast

params = {
    # Model backups
    'load_file': 'saves/approximate',
}

class FinalAgent:

    def __init__(self, player, args):
        self.player = player
        self.weights = {} 
        self.params = params
        self.load()

    def updateRnd(self, rnd) :
        self.ipath = None
        self.t = None
        if self.player.pos == (6, 5):
            self.t = (1, 11)
        elif self.player.pos == (10, 5) :
            self.t = (1, 1)
        elif self.player.pos == (14, 13) :
            self.t = (25, 17)
        elif self.player.pos == (17, 13) :
            self.t = (25, 5)
        self.ipath = []

        self.rnd = rnd
        if self.t:
            self.ipath = astar(self.player.pos, self.t, state.layout.walls, state.layout.width, state.layout.height)
 

    def getLegalActions(self, state) :
        return Actions.getPossibleActions(self.player.pos, state.layout.walls)

    def computeActionFromQValues(self, state) :

        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return Directions.STOP

        QValue = -1e10

        for legalAction in legalActions:
            QValueTemp = self.getQValue(state, legalAction)
            if QValueTemp > QValue:
                action = legalAction
                QValue = QValueTemp
        return action

    def getQValue(self, state, action) :
        QValue = 0.0
        features = util.getFeatures(self.player,state, action)
        for feature in features :
            if not feature in self.weights :
                self.weights[feature] = 0
            QValue += features[feature] * self.weights[feature]
        return QValue

    def init(self, state) :
        self.ipath = None
        self.t = None

        if self.player.pos == (6, 5):
            self.t = (1, 11)
        elif self.player.pos == (10, 5) :
            self.t = (1, 1)
        elif self.player.pos == (14, 13) :
            self.t = (25, 17)
        elif self.player.pos == (17, 13) :
            self.t = (25, 5)
        self.ipath = []

        if self.t:
            self.ipath = astar(self.player.pos, self.t, state.layout.walls, state.layout.width, state.layout.height)


    def getAction(self, state):
        if len(self.ipath) > 0 :
            t = self.ipath[0]

            self.ipath.remove(self.ipath[0])

            action = Actions.vectorToDirection((t[0] - self.player.pos[0], t[1] - self.player.pos[1]))

            return action
        else :

            return self.computeActionFromQValues(state)
            
        return Directions.STOP



    def load(self) :

        if self.params['load_file'] :
            with open(self.params['load_file']) as f:
                s = f.read()
                obj = ast.literal_eval(s)
                self.weights = obj['weights']

            
                

