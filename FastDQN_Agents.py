
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
    'load_file': 'saves/qtable',
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


class FastDQNAgent:

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
    # self.player_hash = {}
    self.load()
  
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
    h = state.hash(action)
    # print(h)
    return 0.0 if h not in self.QValues else self.QValues[h]

  def getAction(self, state) :
    # neighbors = Actions.getLegalNeighbors(self.player.pos, state.layout.walls)
    # foods = [p for p in neighbors if state.food[p[0]][p[1]]]

    legalActions = [a for a in self.getLegalActions(state, self.player.pos)]
    # random.choice(legalActions)
    action = random.choice(legalActions) 
    # if len(foods) > 0 :
    #   p = random.choice(foods)
    #   action = Actions.vectorToDirection((p[0] - self.player.pos[0], p[1] - self.player.pos[1]))


    if np.random.rand() > self.eps :
      action = self.computeActionFromQValues(state) 
    return action
    
  def train(self) :

    smp = random.sample(self.exps, 64)
          
    for (state, newState, pos, newPos, reward) in smp:
      action = Actions.vectorToDirection((newPos[0] - pos[0], newPos[1] - pos[1])) 

      curQValue = self.getQValue(state, action)
      # print('\n-------')
      # print(state)
      # print('->', action)
      # print(newState)

      nextQValue = (1-self.alpha) * curQValue + self.alpha * (reward + self.params['discount'] * self.computeValueFromQValues(newState, newPos) )
      # print('lscore=%2d,nscore=%2d,reward=%2d,curQ=%.2f, nxtQ=%.2f' % 
      #     (state.pacmanScore, newState.pacmanScore,newState.pacmanScore - state.pacmanScore, curQValue, nextQValue))
      # print(action, curQValue, nextQValue, newState.pacmanScore - state.pacmanScore)
      # if reward > 0 and nextQValue < curQValue :
      #   print(self.params['discount'])
      #   print('lscore=%2d,nscore=%2d,reward=%2d,curQ=%.2f, nxtQ=%.2f' % 
      #     (state.pacmanScore, newState.pacmanScore,newState.pacmanScore - state.pacmanScore, curQValue, nextQValue))
      #   raise 'hhh'
      self.QValues[state.hash(action)] = nextQValue
      self.state_hash[state.hash(action)] = 1
      self.eps = max(self.params['eps_final'], 1.00 - float(self.cnt) / float(self.params['eps_step']))
        
    # sys.exit()
  def observationFunction(self, newState) :

      
    pos = self.player.pos
    state = self.last_state


    if self.last_state is not None :


      self.current_score = newState.pacmanScore
      reward = self.current_score - self.last_score
      self.last_score = self.current_score
      self.last_reward = reward
      self.ep_rew += self.last_reward
      self.exps.append((state, newState.copy(), self.last_pos, pos, reward))
      if len(self.exps) > 10000:
        self.exps.popleft()


      self.local_cnt += 1
      if self.local_cnt > 1000 and self.local_cnt % 100 == 0:
        self.train()


      self.won = newState.pacmanScore > newState.ghostScore
    self.last_state = newState.copy()
    self.last_pos = pos
    self.frame += 1

  def init(self,state):
    self.frame = 0
    self.numeps += 1
    self.last_state = state.copy()
    self.last_pos = self.player.pos
    self.won = True


  def final(self, state) :
    self.cnt += 1
    if self.cnt % 10 != 0 :
      return
    # Print stats
    # log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(
    #     self.params['height'])+'-x-'+str(self.params['num_training'])+'.log', 'a')
    # log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
    #                 (self.numeps, self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
    # log_file.write("| Q: %10f | won: %r \n" %
    #                 ((max(self.Q_global, default=float('nan')), self.won)))
    sys.stdout.write("# %4d | steps: %5d | t: %4f | r: %12f | e: %10f | won: %s\n" %
                      (self.numeps, self.local_cnt, time.time()-self.s, self.ep_rew, self.eps, self.won))
    # sys.stdout.write("| Q: %10f | won: %r \n" %
    # print(self.QValues)
    # print(self.state_hash)

    print(' %4d %4d' % (len(self.QValues), len(self.state_hash)))
    sys.stdout.flush()

    if self.cnt > 0 and  self.cnt % 5000 == 0:  
      print('save')
      self.save()
  

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


  
  