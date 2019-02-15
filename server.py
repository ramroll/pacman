from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib
import util
import layout
import sys
import types
import time
import random
import os
from game import Game
from state import State
# import graphicsDisplay

data = {'result': 'this is a test'}
host = ('', 1080)
pixels = []
width = 0
height = 0
args = dict()
agentOpts = dict()
game = 0


def loadAgent(pacman, nographics):
    # Looks through all pythonPath Directories for the right module,
    pythonPathStr = os.path.expandvars("$PYTHONPATH")
    if pythonPathStr.find(';') == -1:
        pythonPathDirs = pythonPathStr.split(':')
    else:
        pythonPathDirs = pythonPathStr.split(';')
    pythonPathDirs.append('.')

    for moduleDir in ['.']:
        if not os.path.isdir(moduleDir):
            continue
        moduleNames = [f for f in os.listdir(
            moduleDir) if f.endswith('gents.py')]
        for modulename in moduleNames:
            # modulename:  pacmanDQN_Agents.py

            try:
                module = __import__(modulename[:-3])
            except ImportError:
                continue
            if pacman in dir(module):
                if nographics and modulename == 'keyboardAgents.py':
                    raise Exception(
                        'Using the keyboard requires graphics (not text display)')
                # Return here
                return getattr(module, pacman)
    raise Exception('The agent ' + pacman +
                    ' is not specified in any *Agents.py.')


class Resquest(BaseHTTPRequestHandler):

    def do_POST(self):
        params = self.rfile.read(
            int(self.headers['content-length']))
        jsonData = json.loads(params)
        global pixels, width, height, args, agentOpts, game
        # 初始化地图
        if 'map' in jsonData:
            pixels = jsonData['map']['pixels']
            width = len(pixels[0])
            height = len(pixels)
            args = dict()
            agentOpts = dict()
            agentOpts['width'] = width
            agentOpts['height'] = height
            agentOpts['numTraining'] = 2950
            args['layout'] = layout.initLayout(pixels)
            args['numTraining'] = 2950
            args['pacmanType'] = loadAgent('FinalAgent', False)
            args['ghostType'] = loadAgent('RandomGhost', False)
            # args['display'] = graphicsDisplay.PacmanGraphics(1.0, 0.1)
            args['numGames'] = 3000
            args['record'] = False
            args['catchExceptions'] = False
            args['timeout'] = 30
            args['agentOpts'] = agentOpts
            game = None
            # game = Game(args)
            # game.notifyAgentHooks('init')
            # self.game = Game(self.args)
            # game.run()
        elif 'pacDots' in jsonData:
            args['pacmanFeast'] = jsonData['pacmanFeast']
            args['layout'] = args['layout'].refreshLayout(jsonData)
            # print(jsonData)
            # game = Game(args)
            if not game:
                game = Game(args)
                game.notifyAgentHooks('init')
            else:
                game.nextRound(args['layout'])
            nextMove = game.hintNextMove()
            # print(game.state)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(nextMove).encode())
            return
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


""" 
    def do_GET(self):
        print('get: ', self.path)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
 """

if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
