
from gameState import GameState
from game import Game
class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout

    def newGame(self, layout, pacmanType, ghostType, display, agentOpts, quiet=False, catchExceptions=False):
        # TODO done
        # agents = [pacmanAgent] + ghostAgents[:layout.getNumGhosts()]
        agents = []
        # agents = pacmanAgent[:layout.numPacman] + \
        #     ghostAgents[:layout.getNumGhosts()]
        initState = GameState()
        initState.initialize(layout)


        for index, agentState in enumerate(initState.data.agentStates): 
            if agentState.isPacman :
                agents.append(pacmanType(agentOpts, index, agentState))
            else :
                agents.append(ghostType(agentOpts, index, agentState))

        game = Game(agents, display, self, catchExceptions=catchExceptions)
        game.state = initState
        initState.game = game


        self.initialState = initState.deepCopy()
  
        self.quiet = quiet
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if state.isWin():
            self.win(state, game)
        if state.isLose():
            self.lose(state, game)

    def win(self, state, game):
        if not self.quiet:
            print(("Pacman emerges victorious! Score: %d" % state.data.score))
        game.gameOver = True

    def lose(self, state, game):
        if not self.quiet:
            print(("Pacman died! Score: %d" % state.data.score))
        game.gameOver = True

    def getProgress(self, game):
        return float(game.state.getNumFood()) / self.initialState.getNumFood()

    def agentCrash(self, game, agentIndex):
        """ 
        TODO agent is pacman if agentIndex == 0
        """
        if agentIndex == 0:
            print("Pacman crashed")
        else:
            print("A ghost crashed")

    def getMaxTotalTime(self, agentIndex):
        return self.timeout

    def getMaxStartupTime(self, agentIndex):
        return self.timeout

    def getMoveWarningTime(self, agentIndex):
        return self.timeout

    def getMoveTimeout(self, agentIndex):
        return self.timeout

    def getMaxTimeWarnings(self, agentIndex):
        return 0

