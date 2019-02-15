
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


def normalize(v):
  if v > 0 : 
    return 1
  elif v < 0 :
    return -1 
  return 0


def hash_d(d):
  if d == (-1,-1) :
    return 1
  elif d == (0, -1):
    return 2
  elif d == (1, -1) :
    return 3
  elif d == (1, 0) :
    return 4
  elif d == (1, 1) :
    return 5
  elif d == (0, 1) :
    return 6
  elif d == (-1, 1):
    return 7
  elif d == (-1, 0):
    return 8
  else:
    return 0
def desc(p0, p1) :
  dist = util.manhattanDistance(p0, p1)
  v = 0
  if dist < 2 :
    v = 1
  elif dist < 4 :
    v = 2
  else:
    v = 3


  vx, vy = (p1[0] - p0[0] , p1[1] - p0[1])
  d = (normalize(vx), normalize(vy))

  return hash_d(d) * v
 
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
    

    self.total_rew = 0
    self.exps = deque()
    # self.player_hash = {}
    self.load()
  
  def getLegalActions(self, state, pos) :
    return Actions.getPossibleActions(pos, state.layout.walls)

  def computeValueFromQValues(self, state, pos, superVal):
      """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
      """
      "*** YOUR CODE HERE ***"
      qvalues = [self.getQValue(state, pos, action, superVal) for action in self.getLegalActions(state, pos)]
      if not len(qvalues): return 0.0
      return max(qvalues)

  def computeActionFromQValues(self, state) :

    legalActions = self.getLegalActions(state, self.player.pos)
    if len(legalActions) == 0:
      return Directions.STOP

    QValue = -1e10

    for legalAction in legalActions:
      QValueTemp = self.getQValue(state, self.player.pos, legalAction, self.player.super)
      if QValueTemp > QValue:
        action = legalAction
        QValue = QValueTemp
    return action
  
  def hash_state(self, state, pos, action, s) :

    h_walls = util.wall_desc(pos, state.layout.walls)


    aliveGhosts = state.aliveGhosts()
    closest2Food = util.closestNFood(pos, state.layout.food, state.layout.walls, 1, aliveGhosts)
    closest2Ghost = util.closest2Ghost(pos, aliveGhosts)

    closestCapsule = None
    if len(state.capsules) :
      closestCapsule = min(state.capsules, key = lambda c : util.manhattanDistance(c, pos))


    h_capsule = 0
    h_ghosts = 0
    h_food = 0
    h_super = 1 if s > 0 else 0
    if closestCapsule :
      h_capsule = desc(pos, closestCapsule) 
    if len(closest2Ghost) == 1:
      h_ghosts = desc( pos, closest2Ghost[0] )
    if len(closest2Ghost) == 2:
      h_ghosts = desc(pos, closest2Ghost[0]) + desc(pos, closest2Ghost[1]) * 31

    if len(closest2Food) == 1:
      h_food = desc(pos, closest2Food[0])
    if len(closest2Food) == 2:
      h_food = desc(pos, closest2Food[0]) +desc(pos, closest2Food[1]) * 31

    h_action = 0
    if action == 'North':
      h_action = 1
    elif action == 'South':
      h_action = 2
    elif action == 'East':
      h_action = 3
    elif action == 'West' :
      h_action = 4
    
    return h_walls * 1e12 + h_ghosts * 1e9 + h_food * 1e6 + h_super * 1e3 + h_action


  def getQValue(self, state, pos, action, superVal) :
    # h = state.hash(action)
    h = self.hash_state(state, pos, action, superVal)
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
          
    for (state, newState, pos, newPos, reward, superVal) in smp:
      action = Actions.vectorToDirection((newPos[0] - pos[0], newPos[1] - pos[1])) 

      curQValue = self.getQValue(state, pos, action, superVal)
      # print('\n-------')
      # print(state)
      # print('->', action)
      # print(newState)

      nextQValue = (1-self.alpha) * curQValue + self.alpha * (reward + self.params['discount'] * self.computeValueFromQValues(newState, newPos, superVal) )
      # print('lscore=%2d,nscore=%2d,reward=%2d,curQ=%.2f, nxtQ=%.2f' % 
      #     (state.pacmanScore, newState.pacmanScore,newState.pacmanScore - state.pacmanScore, curQValue, nextQValue))
      # print(action, curQValue, nextQValue, newState.pacmanScore - state.pacmanScore)
      # if reward > 0 and nextQValue < curQValue :
      #   print(self.params['discount'])
      #   print('lscore=%2d,nscore=%2d,reward=%2d,curQ=%.2f, nxtQ=%.2f' % 
      #     (state.pacmanScore, newState.pacmanScore,newState.pacmanScore - state.pacmanScore, curQValue, nextQValue))
      #   raise 'hhh'
      self.QValues[self.hash_state(state, pos, action,superVal)] = nextQValue
      # self.state_hash[state.hash(action)] = 1
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
      self.exps.append((state, newState.copy(), self.last_pos, pos, reward, self.player.super))
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
    # Print stats
    # log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(
    #     self.params['height'])+'-x-'+str(self.params['num_training'])+'.log', 'a')
    # log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
    #                 (self.numeps, self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
    # log_file.write("| Q: %10f | won: %r \n" %
    #                 ((max(self.Q_global, default=float('nan')), self.won)))
    self.total_rew += self.ep_rew
    
    if self.cnt % 40 == 0 :
      sys.stdout.write("# %4d | steps: %5d | hsize: %4d | avgr: %6.2f | r : %4d | e: %10f | won: %s\n" %
                        (self.numeps, self.local_cnt, len(self.QValues), self.total_rew / self.cnt,  self.ep_rew,  self.eps, self.won))

      # print(' %4d %4d' % (len(self.QValues), len(self.state_hash)))
      sys.stdout.flush()

    if self.cnt > 0 and  self.cnt % 500 == 0:  
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
        self.eps = .5
        self.cnt = 0
        self.numeps = 0
        self.total_rew = 0
        print('loaded')
        


  
  