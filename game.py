from os import system, name
from player import Player
from state import State
import time
import util
import random
from layout import Layout
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
        self.change = 0
        layout = params['layout']

    
    def nextRound(self, layout):
        newState = State(layout)

        ghost_pos = []
        ghosts = []
        players = [p for p in self.players if not p.isPacman]
        for isPacman, pos, super in layout.agentPositions:
            if not isPacman:
                players.append(Player(False, (int(pos[0]), int(pos[1])), 0, 0))
        
        self.players = players
        self.state.players = players



    def createPlayers(self):
        players = []
        agentParams = {
            'num_training': self.params['numTraining'],
            'layout': self.params['layout']
        }

        i = 0
        for isPacman, pos, super in self.state.layout.agentPositions:
            player = Player(isPacman, (int(pos[0]), int(pos[1])), i, super)
            i += 1
            player.setAgent(
                self.pacmanType if isPacman else self.ghostType, self.state, agentParams)
            players.append(player)
        # print('createplayers end...')
        return players

    def initialize(self, layout):
        self.state = State(layout)
        self.layout = layout
        self.players = self.createPlayers()
        # print('l....', len(self.players))
        self.state.setPlayers(self.players)

    # 游戏循环

    def run(self):

        rnd = 1
        for i in range(self.params['numGames']):
            self.round(rnd)
            rnd += 1

    # 每次回合刷新

    def refresh(self, rnd):
        # rnd = 1000
        state = self.state
        p_set = set()
        RND_NUM = 2000
        if rnd % RND_NUM == 0:
            

            # 模拟旋转
            # rotate = (random.random())
            layoutText = state.layout.layoutText
            
            height = len(layoutText)
            width = len(layoutText[0])
            newLayout = []

            # flip-x
            if (rnd // RND_NUM) % 10 in [1,2,6,7]:
                print('flip-x')
                for i in range(height):
                    row = ['' for x in range(width)]
                    for j in range(width):
                        # print(layoutText[i], j)
                        row.append(layoutText[i][-1-j])
                    newLayout.append(row)
                newLayoutText = [''.join(p) for p in newLayout]
            # flip-y
            if (rnd // RND_NUM) % 10 in [3,4,8,9]:
                print('flip-y')
                for i in range(height):
                    row = ['' for x in range(width)]
                    for j in range(width):
                        row.append(layoutText[-1-i][j])
                    newLayout.append(row)
                newLayoutText = [''.join(p) for p in newLayout]
            # rotate
            if (rnd // RND_NUM) % 10 in [5, 0]:    
                print('rotate')
                for i in range(width):
                    row = ['' for x in range(height)]
                    for j in range(height):
                        row.append(layoutText[j][i])
                    newLayout.append(row)
                newLayoutText = [''.join(p) for p in newLayout]
            
            # 随机生成pacman和Ghost
            # 豆子和空位置的比例
            ratio = 0.2 + 0.8*random.random()

            for i in range(len(newLayoutText)) :
                for j in range(len(newLayoutText[i])):
                    c = newLayoutText[i][j]
                    r = random.random()
                    if c == '.' or c == ' ' or c == 'o' or c == 'P' or c == 'G' :
                        if r < ratio :
                            newLayoutText[i] = newLayoutText[i][:j] + '.' + newLayoutText[i][j+1:]
                        else:
                            newLayoutText[i] = newLayoutText[i][:j] + ' ' + newLayoutText[i][j+1:]
            
            
            posList = []
            for i in range(len(newLayoutText)) :
                for j in range(len(newLayoutText[i])):
                    c = newLayoutText[i][j]
                    if c != '%':
                        posList.append((i, j))
            
            # print(len(posList), len(self.players))
            capsuleNum = round( random.random() * 2)
            smp = random.sample(posList, len(self.players) + capsuleNum)
            agents = []
            for i in range(len(smp)) :
                y,x = smp[i]
                c = ''
                if i < len(self.players) :
                    self.players[i].setInitialPos( (x, len(newLayoutText) - y - 1) )
                    self.players[i].refresh()


                    if 'refresh' in dir(self.players[i].agent):
                        self.players[i].agent.refresh()
                    c = 'P' if self.players[i].isPacman else 'G'
                    # agents.append(self.players)
                else :
                    c = 'o'

                newLayoutText[y] = newLayoutText[y][:x] + c + newLayoutText[y][x+1:]
            newLayout = Layout(newLayoutText)
            state.layout = newLayout
            state.refresh()
            print(state)
            # self.initialize(Layout(newLayoutText))

        else:
            for player in self.players:
                player.refresh()
            state.refresh()

           





                        


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
        # print('hintNextMove...')print
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

            if player.isPacman:
                action = player.getAction(state)
                state.next(player, action)
                dirs[pacmanCount] = direct[action]
                pacmanCount += 1
        return {'dirs': dirs}

    # 每个回合
    def round(self, rnd):

        result = 0
        self.refresh(rnd)
        state = self.state
        eps = 0

        if self.params['numTraining'] < rnd:

            self.display = Display()
            self.display.initialize(state)
        self.notifyAgentHooks('init')

        turn = 0
        while(True):

            players = None

            if turn % 2 == 0 :
                players = state.alivePacmans()
            else :
                players = state.aliveGhosts()
            


            for player in players:
                action = player.getAction(state)

                state.next(player, action)

                result = state.whoWins()
                if result != 0:
                    break

            if result != 0:
                self.notifyAgentHooks('observationFunction')
                break
            if turn % 2 == 1:
                self.notifyAgentHooks('observationFunction')  
                if self.params['numTraining'] < rnd:
                    self.display.update(state, rnd)

            eps += 1
            turn += 1
        # self.notifyAgentHooks('observationFunction')
        self.notifyAgentHooks('final')
        R = max(rnd, 100)
        self.rate = 1 - (state.food.count() /
                         state.total_food) if state.total_food > 0 else 0
        self.avg_rate = self.avg_rate * (R-1) / (R) + self.rate / R
        # if rnd % 10 == 0:
        #     self.printResult(rnd, result, state, eps)


class Display:

    def initialize(self, state):
        print(state)

    def update(self, state, rnd):
        system('clear')
        print('round:', rnd)
        print(state)
        time.sleep(0.3)
