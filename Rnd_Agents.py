import random
from actions import Actions
class RndAgent : 

  def __init__(self,player,args) :
    self.player = player

  def getAction(self,state):
    # if not self.player.isPacman :
      # return 'Stop' 
    legal=Actions.getPossibleActions(self.player.pos, state.layout.walls)
    # print(legal)
    return random.choice(legal)
    
    

