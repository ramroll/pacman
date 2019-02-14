
import random
from directions import Directions
from actions import Actions
import util
import time
import numpy as np
import sys
import ast
from collections import deque
params = {
    # Model backups
    'load_file': 'saves/qtable1',
    'save_file': 'saves/qtable',
    'save_interval': 20000,

    # Training parameters
    'train_start': 5000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0004,            # Learning reate
    # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
    # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}


class FastDQNAgent1:

  def __init__(self, player, args) :
    self.player = player
    self.alpha = 0.001
    self.params = params
    self.params.update(args)
    self.cnt = 0

    self.last_reward = 0
    self.last_action = None
    self.current_score = 0
    self.last_score = 0
    self.local_cnt = 0
    self.s = time.time()
    self.numeps = 0
    self.ep_rew = 0
    self.eps = params['eps']
    self.last_state = None
    self.last_pos = self.player.pos
    self.QValues = {}
    self.state_hash = {}

    self.exps = deque()
    self.load()
    # self.player_hash = {}
    # self.load()
  
  def getLegalActions(self, state, pos) :
    return Actions.getPossibleActions(pos, state.layout.walls)

  def computeValueFromQValues(self, state, pos):
      """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
      """
      "*** YOUR CODE HERE ***"
      qvalues = [self.getQValue(state, action) for action in self.getLegalActions(state, pos)]
      if not len(qvalues): return 0.0
      return max(qvalues)

  def computeActionFromQValues(self, state) :

    legalActions = self.getLegalActions(state, self.player.pos)
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
    h = state.hash1(action, self.player)
    # print(h)
    return 0.0 if h not in self.QValues else self.QValues[h]

  def getAction(self, state) :
    legalActions = [a for a in self.getLegalActions(state, self.player.pos)]
    action = self.computeActionFromQValues(state) 
    return action
    



  def save(self) :
    if self.params['save_file'] :
      with open(self.params['save_file'], 'w+') as f : 
        f.write(repr({
          "qtable" : self.QValues,
          "eps" : self.eps,
          "cnt" : self.cnt,
          'numeps' : self.numeps
        }))


  def load(self) :

    if self.params['load_file'] :
      with open(self.params['load_file']) as f:
        s = f.read()

        obj = ast.literal_eval(s)

        self.QValues = obj['qtable']
        self.eps = obj['eps']
        self.cnt = obj['cnt']
        self.numeps = obj['numeps']



  
  