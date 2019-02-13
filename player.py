
class Player:
    def __init__(self, isPacman, pos, index, super=0):
        self.isPacman = isPacman
        self.pos = pos
        self.__pos = pos
        self.alive = True
        self.super = super
        # self.scaredTimer = 0
        self.moves = 0
        self.index = index
        self.dir = None

    def copy(self):
        player = Player(self.isPacman, self.pos, self.index, self.super)
        return player
    def setAgent(self, agentType, state, agentParams):
        self.agent = agentType(self, agentParams)

    def getAction(self, state):
        self.moves += 1
        if self.super > 0:
            self.super -= 1
        return self.agent.getAction(state)

    def refresh(self):
        self.pos = self.__pos
        self.alive = True
        self.moves = 0

    def __hash__(self) :
        return hash(hash(self.pos) + 13 * hash((self.isPacman, self.super)))


