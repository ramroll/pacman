# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html


import numpy as np
import random
import util
import time
import sys

# Pacman game
from directions import Directions

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
from DQN import *

from actions import Actions

params = {
    # Model backups
    'load_file': 'saves/smart',
    'save_file': 'smart',
    'save_interval': 20000,

    # Training parameters
    'train_start': 5000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
    # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}


SIGHT=7
class SmartDQNAgent:
    def __init__(self, player, args):
        self.player = player
        print("Initialise DQN Agent")

        self.params =  params 
        self.params['width'] = SIGHT
        self.params['height'] = SIGHT
        self.params['num_training'] = args['num_training']

        self.params.update(args)
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.qnet = DQN(self.params)

        # time started
        self.general_record_time = time.strftime(
            "%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

    def getMove(self, state):
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict={self.qnet.x: np.reshape(self.current_state,
                                                   (1, SIGHT, SIGHT, 6)),
                           self.qnet.q_t: np.zeros(1),
                           self.qnet.actions: np.zeros((1, 4)),
                           self.qnet.terminals: np.zeros(1),
                           self.qnet.rewards: np.zeros(1)})[0]

            self.Q_global.append(max(self.Q_pred))
            legalActions=[ 
                x for x in Actions.getPossibleActions(self.player.pos, state.layout.walls)
            ]
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(
                    a_winner[0][0])
            if not move in legalActions:
                move = random.choice(legalActions)
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST

    def observation_step(self, state):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            # Process current experience reward
            self.current_score = state.pacmanScore
            reward = self.current_score - self.last_score
            self.last_score = self.current_score
            self.won = (state.pacmanScore > state.ghostScore)
            self.last_reward = reward
            # if reward > 20:
            #     self.last_reward = 50.    # Eat ghost   (Yum! Yum!)
            # elif reward > 0:
            #     self.last_reward = 10.    # Eat food    (Yum!)
            # elif reward < -10:
            #     self.last_reward = -500.  # Get eaten   (Ouch!) -500
            #     self.won = False
            # elif reward < 0:
            #     self.last_reward = -1.    # Punish time (Pff..)

            # if(self.terminal and self.won):
            #     self.last_reward = 100.
            self.ep_rew += self.last_reward

            # Store last experience into memory
            experience = (self.last_state, float(self.last_reward),
                          self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if(params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    # self.qnet.save_ckpt(
                        # 'saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))

                    self.qnet.save_ckpt(
                        'saves/' + params['save_file'])
                    print('Model saved')
                    # import sys
                    # sys.exit()

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt) / float(self.params['eps_step']))

    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(
            self.params['height'])+'-x-'+str(self.params['num_training'])+'.log', 'a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                       (self.numeps, self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" %
                       ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps, self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" %
                         ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = []  # States (s)
            batch_r = []  # Rewards (r)
            batch_a = []  # Actions (a)
            batch_n = []  # Next states (s')
            batch_t = []  # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(
                batch_s, batch_a, batch_t, batch_n, batch_r)

    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total


    def getStateMatrices(self, state):

        to_viewport = change_axis(self.player.pos)
        to_map = change_axis((-self.player.pos[0], -self.player.pos[1]))
        """ Return wall, ghosts, food, capsules matrices """
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = SIGHT, SIGHT 
            grid = state.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for y in range(height):
                for x in range(width) :
                    # (j, i) 对应在屏幕上的坐标
                    sx, sy = to_map((x, y))
                    if not bounded((sx, sy), state.layout.width, state.layout.height): 
                        cell = 1
                    elif grid[-1-sx][sy]:
                        cell = 1
                    else : 
                        cell = 0
                    matrix[x][y] = cell
            return matrix



        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width,height=SIGHT,SIGHT
            matrix = np.zeros((height, width), dtype=np.int8)
            matrix[SIGHT // 2][SIGHT // 2] = 1 
            # matrix[-1-int(pos[1])][int(pos[0])] = cell
            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            matrix = np.zeros((SIGHT, SIGHT), dtype=np.int8)
            for player in state.players:
                if not player.isPacman:
                    if self.player.super == 0:
                        vx, vy = distance_vector(self.player.pos, player.pos)
                        matrix[vx][vy] += 1


            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            matrix = np.zeros((SIGHT, SIGHT), dtype=np.int8)
            for player in state.players:
                if not player.isPacman:
                    if self.player.super > 0:
                        vx, vy = distance_vector(self.player.pos, player.pos)
                        # 需要写入的坐标位置
                        matrix[vx][vy] += 1


            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            grid = state.food
            matrix = np.zeros((SIGHT, SIGHT), dtype=np.int8)
            for y in range(state.layout.height):
                for x in range(state.layout.width):
                    if grid[-1-x][y] :
                        vx, vy = distance_vector(self.player.pos, (x, y))
                        matrix[vx][vy] += 1

            # print(matrix)
            # import sys
            # sys.exit()
            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            # width, height = state.layout.width, state.layout.height
            width, height = SIGHT, SIGHT
            capsules = state.capsules
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in capsules:
                x,y=i
                vx,vy = distance_vector(self.player.pos, (x,y))
                matrix[vx][vy] += 1
            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.layout.width, state.layout.height
        # width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, SIGHT, SIGHT))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation

    def init(self, state):  # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)
        legal=Actions.getPossibleActions(self.player.pos, state.layout.walls)
        if move not in legal:
            move = Directions.STOP
        return move


def to_normal_axis(p, max):
    return (p[0], max - p[1] - 1)
def to_screen_axis(p, max) :
    return (p[0], max - p[1] - 1)

def bounded(p, w, h) :
    return p[0] >= 0 and p[1] >=0 and p[0] < w and p[1] < h

def to_point_in_map(pos, rpos, sight):
    d = sight // 2
    return (pos[0] + rpos[0] - d, pos[1] + rpos[1] - d)

def to_point_in_viewport(pos, gpos, sight) :
    d = sight // 2
    return (int(gpos[0] - (pos[0] - d)), int(gpos[1] - (pos[1] - d)))


# 变换坐标系函数
# 将源点变成t
def change_axis(o) :
    dx, dy = -o[0], -o[1]
    def inner(p) :
        return p[0] + dx, p[1] + dy
    return inner


def distance_vector(origin, target):
    ox,oy = origin
    tx,ty = target
    vx, vy = tx - ox, ty - oy

    d = SIGHT // 2
    if abs(vx) > d:
        vx = d * (1 if vx > 0 else -1)
    if abs(vy) > d:
        vy = d * (1 if vy > 0 else -1)
    return vx, vy