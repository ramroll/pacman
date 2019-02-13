
import random
from directions import Directions
from actions import Actions
import util
import time
import numpy as np
import sys
import ast

params = {
    # Model backups
    'load_file': 'saves/approximate',
    'save_file': 'saves/approximate',
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
    self.won = True
    self.last_state = None
    self.last_pos = self.player.pos
    self.QValues = util.Counter()
    # self.load()
  
  def getLegalActions(self, state) :
    return Actions.getPossibleActions(self.player.pos, state.layout.walls)

  def computeValueFromQValues(self, state):
      """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
      """
      "*** YOUR CODE HERE ***"
      qvalues = [self.getQValue(state, action) for action in self.getLegalActions(state)]
      if not len(qvalues): return 0.0
      return max(qvalues)

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
    return self.QValues[(state, action)]

  def getAction(self, state) :
    legalActions = self.getLegalActions(state)
    action = random.choice(legalActions) 
    if len(legalActions) == 0:
      return action

    if np.random.rand() > self.eps :
      action = self.computeActionFromQValues(state) 
    return action
    
    
    
  def observationFunction(self, newState) :


      
    pos = self.player.pos
    state = self.last_state
    action = Actions.vectorToDirection((pos[0] - self.last_pos[0], pos[1] - self.last_pos[1])) 

    self.last_state = newState
    self.last_pos = pos
    if self.last_action is not None :
      curQValue = self.getQValue(state, action)

      self.current_score = state.pacmanScore
      reward = self.current_score - self.last_score
      self.last_score = self.current_score
      self.last_reward = reward
      self.ep_rew += self.last_reward


      self.QValues[(state, action)] = (1-self.alpha) * curQValue + self.alpha * (reward \
                                          + self.params['discount'] * self.computeValueFromQValues(newState) )
      self.local_cnt += 1
      self.eps = max(self.params['eps_final'], 1.00 - float(self.cnt) / float(self.params['eps_step']))
      self.won = state.pacmanScore > state.ghostScore
    self.last_action = action

  def init(self,state):
    self.frame = 0
    self.numeps += 1
    self.last_state = state
    self.last_pos = self.player.pos


  def final(self, state) :
    self.cnt += 1

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
    print(' %4d' % (len(self.QValues)))
    sys.stdout.flush()

    if self.cnt > 0 and  self.cnt % 5000 == 0:  
      print('save')
      # self.save()
  

  def save(self) :
    if self.params['save_file'] :
      with open(self.params['save_file'], 'w+') as f : 
        print(self.weights)
        f.write(repr({
          "weights" : self.weights,
          "eps" : self.eps,
          "cnt" : self.cnt,
          'numeps' : self.numeps
        }))


  def load(self) :
    try:
      if self.params['load_file'] :
        with open(self.params['load_file']) as f:
          s = f.read()
          obj = ast.literal_eval(s)
          self.weights = obj['weights']
          self.eps = obj.eps
          
    
    except :
      return {} 


  
  