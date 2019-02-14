from actions import Actions
from grid import Grid

TIME_PENALTY = -1


class State:
    def __init__(self, layout):
        self.layout = layout.deepCopy()
        self.food = layout.food.copy()
        self.capsules = layout.capsules.copy()
        self.pacmanScore = 0
        self.ghostScore = layout.food.count() * 10

        # For graphics
        self._foodEaten = None
        self._capsuleEaten = None

    def refresh(self):
        self.food = self.layout.food.copy()
        self.pacmanScore = 0
        self.ghostScore = self.layout.food.count() * 10
        self.total_food = self.food.count()

        self.capsules = self.layout.capsules.copy()

    def copy(self):
        state = State(self.layout)
        state.pacmanScore = self.pacmanScore
        state.ghostScore = self.ghostScore
        state.food = self.food.copy()
        state.capsules = self.capsules.copy()

        state.players = []
        for player in self.players:
            state.players.append(player.copy())
        return state

    def setPlayers(self, players):
        self.players = players

    def whoWins(self):
        p = len(self.alivePacmans())
        g = len(self.aliveGhosts())
        if p == 0:
            return 1
        elif g == 0:
            return 2
        return 0

    def alivePlayers(self):
        return [x for x in self.players if x.alive == True]

    def aliveGhosts(self):
        return [x for x in self.players if (x.isPacman == False) and x.alive == True]

    def alivePacmans(self):
        return [x for x in self.players if x.isPacman and x.alive == True]

    def next(self, player, action):
        # 检查是否合法
        Actions.checkLegal(player.pos, action, self.layout.walls)
        # 更新坐标
        (player.pos, player.dir) = Actions.getSuccessor(player.pos, action)
        # 更新分数和生死状态
        self.calculateScore(player)

    def calculateScore(self, player):
        x, y = player.pos
        numFood = self.food.count()

        if player.isPacman:
            # if player.moves > 50 :
            #   time_p = 2
            # elif player.moves > 100 :
            #   time_p = 3

            if player.moves > 100:
                player.alive = False
                return

            # self.pacmanScore += TIME_PENALTY
            g = self.aliveGhosts()
            collision = [x for x in g if x.pos == player.pos]
            if len(collision) > 0:
                if player.super > 0:
                    for p in collision:
                        p.alive = False
                        # self.pacmanScore += 500
                        # self.ghostScore -= 500
                else:
                    player.alive = False
                    # self.pacmanScore -= 500
                    # self.ghostScore += 500
                return

            if (x, y) in self.capsules:
                self.capsules.remove((x, y))
                player.super = 5 

            if self.food[x][y]:
                self.pacmanScore += 10
                self.ghostScore -= 10
                self.food[x][y] = False
                self._foodEaten = (x, y)
                numFood -= 1
                if(numFood == 0):
                    # self.pacmanScore += 500
                    # self.ghostScore -= 500
                    for ghost in self.aliveGhosts():
                        ghost.alive = False
        else:
            g = self.alivePacmans()
            # self.ghostScore += TIME_PENALTY
            eaten = [x for x in g if x.pos == player.pos]
            # print(eaten)
            for pacman in eaten:
                if pacman.super > 0:
                    player.alive = False
                    # self.pacmanScore += 500
                    # self.ghostScore -= 500
                else:
                    pacman.alive = False
                    # self.ghostScore += 500
                    # self.pacmanScore -= 500

    def getPacmanPositions(self):
        alivePacmans = self.alivePacmans()
        return [x.pos for x in alivePacmans]

    def hash(self, action):
        hash_food = hash(self.food)
        hash_players = '-'.join([x.hash() for x in self.players])

        return 'f{0}p{1}|{2}'.format(hash_food, hash_players, action)

    def hash1(self, action, player):
        hash_food = hash(self.food)
        hash_players = '-'.join([x.hash() for x in self.players if x.index ==
                                 player.index or x.isPacman is not True])
        return 'f{0}p{1}|{2}'.format(hash_food, hash_players, action)

    def __str__(self):
        width, height = self.layout.width, self.layout.height
        map = Grid(width, height)
        if isinstance(self.food, type((1, 2))):
            self.food = reconstituteGrid(self.food)
        for x in range(width):
            for y in range(height):
                food, walls = self.food, self.layout.walls
                map[x][y] = self._foodWallStr(food[x][y], walls[x][y])
        for x, y in self.capsules:
            map[x][y] = 'o'

        for player in self.players:
            x, y = player.pos
            if player.isPacman:
                map[x][y] = self._pacStr()
            else:
                map[x][y] = self._ghostStr()

        return str(map)

    def _foodWallStr(self, hasFood, hasWall):
        if hasFood:
            return '.'
        elif hasWall:
            return '%'
        else:
            return ' '

    def _pacStr(self):
        return 'P'

    def _ghostStr(self):
        return 'G'
