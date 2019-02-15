
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
    
    self.avgr = 0.0
    self.exps = deque()
    self.wonr = 0.0
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
  
  
  def hash_dir(self, v) :
    x,y = v
    if x == 1 and y == 0 :
      return 0
    elif x == -1 and y == 0:
      return 1
    elif x == 0 and y == 1:
      return 2
    else:
      return 3

  def ghost_desc(self, pos, walls, aliveGhosts):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    parent = {}

    gset = set()

    for ghost in aliveGhosts:
      gset.add(ghost.pos)

    ghosts = []
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
          continue
        if dist > 5 :
          continue
          
        expanded.add((pos_x, pos_y))

        if (pos_x, pos_y) in gset:
          ghosts.append((pos_x, pos_y, dist))

        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
          if (nbr_x, nbr_y) not in expanded:
            parent[(nbr_x, nbr_y)] = (pos_x, pos_y)
          fringe.append((nbr_x, nbr_y, dist+1))

    h_list = [0,0,0,0]
    for ghost in ghosts:
      gx,gy,dist = ghost
      if (gx,gy) == pos:
        continue

      path = []
      p = (gx, gy)
      while p in parent:
        path.append(p)
        p = parent[p]
      path.reverse()
      x = path[0] 
      v = (x[0] - pos[0], x[1] - pos[1]) 
      h_list[self.hash_dir(v)] = dist 
    
    base = 1
    h = 0
    for i in h_list:
      if i:
        h += i * base
      base *= 10
    return h
      
      
    
      
  def food_desc(self,pos,walls,foods,ghosts) :
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    parent = {}

    gset = [g.pos for g in ghosts]
    food_pos = None
    m = 0
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
          continue

        expanded.add((pos_x, pos_y))

        if foods[pos_x][pos_y] > 0 :
          food_pos = (pos_x, pos_y)
          m = dist
          break

        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:

          for (gx, gy) in gset:
            if util.manhattanDistance((gx, gy), pos) < dist + 1 : 
              continue
          if (nbr_x, nbr_y) in gset:
            continue
          if (nbr_x, nbr_y) not in expanded :
            parent[(nbr_x, nbr_y)] = (pos_x, pos_y)
          fringe.append((nbr_x, nbr_y, dist+1))

    if food_pos and food_pos != pos:

      path = []
      p = food_pos
      while p in parent:
        path.append(p)
        p = parent[p]
      path.reverse()
      x = path[0]
      v = (x[0] - pos[0], x[1] - pos[1])

      h = dist * 100 + self.hash_dir(v)
      return h
    return 0

  def hash_state(self, state, pos, action, s) :

    
    # h_walls = util.wall_desc(pos, state.layout.walls)
    aliveGhosts = state.aliveGhosts()

    h_food = self.food_desc(pos, state.layout.walls, state.food, aliveGhosts)
    h_ghosts = self.ghost_desc(pos, state.layout.walls, aliveGhosts)


    closestCapsule = None
    if len(state.capsules) :
      closestCapsule = min(state.capsules, key = lambda c : util.manhattanDistance(c, pos))

    h_action = 0
    if action == 'North':
      h_action = 1
    elif action == 'South':
      h_action = 2
    elif action == 'East':
      h_action = 3
    elif action == 'West' :
      h_action = 4
    
    h_super = 1 if s > 0 else 0
    return h_ghosts * 1e9 + h_food * 1e3 + h_super * 1e1 + h_action


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


    if self.params['num_training'] == 0:
      return self.computeActionFromQValues(state) 
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

    if self.params['num_training'] == 0:
      return

      
    pos = self.player.pos
    state = self.last_state


    if self.last_state is not None :


      self.current_score = newState.pacmanScore
      reward = self.current_score - self.last_score
      self.last_score = self.current_score
      self.last_reward = reward
      self.ep_rew += self.last_reward
      self.exps.append((state.copy(), newState.copy(), self.last_pos, pos, reward, self.player.super))
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

    if self.params['num_training'] == 0:
      return
    self.cnt += 1
    # Print stats
    # log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(
    #     self.params['height'])+'-x-'+str(self.params['num_training'])+'.log', 'a')
    # log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
    #                 (self.numeps, self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
    # log_file.write("| Q: %10f | won: %r \n" %
    #                 ((max(self.Q_global, default=float('nan')), self.won)))
    self.avgr = self.ep_rew * 0.01 + self.avgr * 0.99
    self.wonr = (0.001 if self.won else 0) + self.wonr * 0.999
    
    if self.cnt % 40 == 0 :
      sys.stdout.write("# %4d | steps: %5d | hsize: %4d | avgr: %6.2f | wonr : %6.2f | r : %4d | e: %10f | won: %s\n" %
                        (self.numeps, self.local_cnt, len(self.QValues), self.avgr, self.wonr, self.ep_rew,  self.eps, self.won))

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
      try:
        with open(self.params['load_file']) as f:
          s = f.read()

          obj = ast.literal_eval(s)

          self.QValues = obj['qtable']
          self.eps = .5
          self.cnt = obj['cnt'] 
          self.numeps = 0
          print('loaded')
      except :
        return
        


  
  