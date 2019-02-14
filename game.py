from os import system, name
from player import Player
from state import State
import time
# from graphicsUtils import *


class Game:

    def __init__(self, params):
        self.params = params
        self.pacmanType = params['pacmanType']
        self.ghostType = params['ghostType']
        self.initialize(params['layout'])
        # self.display = params['display']

        # 没回合吃掉豆子的比例
        self.rate = 0
        # 平均比例
        self.avg_rate = 0

    def createPlayers(self):
        players = []
        agentParams = {
            'num_training': self.params['numTraining'],
            'layout': self.params['layout']
        }

        i = 0
        for isPacman, pos, super in self.layout.agentPositions:
            player = Player(isPacman, (int(pos[0]), int(pos[1])), i, super)
            i += 1
            player.setAgent(
                self.pacmanType if isPacman else self.ghostType, self.state, agentParams)
            players.append(player)
        return players

    def initialize(self, layout):
        self.state = State(layout)
        self.layout = layout
        self.players = self.createPlayers()
        self.state.setPlayers(self.players)

    # 游戏循环

    def run(self):

        rnd = 1
        for i in range(self.params['numGames']):
            self.round(rnd)
            rnd += 1

    # 每次回合刷新

    def refresh(self):
        self.state.refresh()
        for player in self.players:
            player.refresh()

    # 打印游戏结果

    def printResult(self, rnd, result, state, eps):

        pacmanWin = state.pacmanScore > state.ghostScore
        if not pacmanWin:
            print('{0}, ghost win, pacman({1}), ghost({2}), eps={3}, rate={4:.2f}'.format(
                rnd, self.state.pacmanScore, self.state.ghostScore, eps, self.rate))
        else:
            print('{0}, pacman win, pacman({1}), ghost({2}), eps={3}, rate={4:.2f}'.format(
                rnd, self.state.pacmanScore, self.state.ghostScore, eps, self.rate))
        print('avg_rate : %.3f' % self.avg_rate)

    # 原来调用Agent的回调方法

    def notifyAgentHooks(self, name):
        for player in self.players:
            if name in dir(player.agent):
                method_to_call = getattr(player.agent, name)
                method_to_call(self.state)
                # player.agent[name](self.state)

    # 提示下一步

    def hintNextMove(self):
        state = self.state
        alivePlayers = state.alivePlayers()
        pacmanCount = 0
        direct = {
            'North': 'UP',
            'South': 'DOWN',
            'East': 'RIGHT',
            'West': 'LEFT',
            'Stop': 'STOP'
        }
        dirs = dict()
        for player in alivePlayers:
            if not player.alive:
                continue
            action = player.getAction(state)
            if player.isPacman:
                dirs[pacmanCount] = direct[action]
                pacmanCount += 1
        return {'dirs': dirs}

    # 每个回合
    def round(self, rnd):

        result = 0
        self.refresh()
        state = self.state
        eps = 0

        if self.params['numTraining'] < rnd:

            self.display = Display()
            self.display.initialize(state)
        self.notifyAgentHooks('init')
        while(result == 0):

            alivePlayers = state.alivePlayers()
            for player in alivePlayers:
                
                if not player.alive:
                    continue
                action = player.getAction(state)

                state.next(player, action)
                if(player.pos[0] < 0 and player.pos[1] < 0) :
                    print(player.isPacman, player.pos, action)
                    import sys
                    sys.exit()
                result = state.whoWins()

                if result != 0:
                    break
            if self.params['numTraining'] < rnd:
                self.display.update(state, rnd)

            self.notifyAgentHooks('observationFunction')
            eps += 1

        self.notifyAgentHooks('final')
        R = max(rnd, 100)
        self.rate = 1 - (state.food.count() /
                         state.total_food) if state.total_food > 0 else 0
        self.avg_rate = self.avg_rate * (R-1) / (R) + self.rate / R 
        if rnd % 10 == 0 :
            self.printResult(rnd, result, state, eps)


class Display:

    def initialize(self, state):
        print(state)

    def update(self, state, rnd):
        system('clear')
        print('round:', rnd)
        print(state)
        time.sleep(0.3)
