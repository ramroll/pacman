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



class SmartDQN1Agent:
    def __init__(self, player, args):
        self.player = player
        print("Initialise DQN Agent")

        self.params =  params 
        self.params['num_training'] = args['num_training']

        self.params.update(args)
        self.params['width'] = self.params['layout'].width
        self.params['height'] = self.params['layout'].height
        self.save_file = self.params['save_file'] + '-' + str(player.index)
        self.load_file = self.params['load_file'] + '-' + str(player.index)
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        dqnParams = self.params.copy()
        dqnParams['load_file'] = self.load_file
        self.qnet = DQN(dqnParams)

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
        self.Q_pred = self.qnet.sess.run(
            self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                                    (1, self.params['width'], self.params['height'], 6)), 
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
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.layout.width, state.layout.height
            grid = state.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.layout.width, state.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for player in state.players:
                if player.isPacman and player != self.player:
                    pos = player.pos 
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.layout.width, state.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for player in state.players:
                if not player.isPacman:
                    if not player.super > 0:
                        pos = player.pos 
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.layout.width, state.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for player in state.players:
                if not player.isPacman:
                    if player.super > 0:
                        pos = player.pos 
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.layout.width, state.layout.height
            grid = state.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.layout.width, state.layout.height
            capsules = state.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.layout.width, state.layout.height 
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

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


# 变换坐标系函数
# 将源点变成t
def change_axis(o) :
    dx, dy = -o[0], -o[1]
    def inner(p) :
        return p[0] + dx, p[1] + dy
    return inner

