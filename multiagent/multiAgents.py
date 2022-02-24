# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from hashlib import new
from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        # -1 for each action
        evaluationScore = -1

        # matrix of game map
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # print("Succesor Game State: ",successorGameState)

        # (x,y) - position in game
        newPos = successorGameState.getPacmanPosition()
        #print("Sucessor New Postion: ", newPos)

        # matrix of T/F
        currFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        #print("Sucessor New Food: ", newFood)

        # [array of ghost states for this sucessor]
        newGhostStates = successorGameState.getGhostStates()
        # print("New Ghost States: ", str(newGhostStates))
        # ((x,y), Direction)

        # food distance to player evaluation
        dist = 0
        reward = -0.1
        for food in newFood.asList():
            dist += util.manhattanDistance(newPos, food)

        evaluationScore += dist * reward


         # pellet distance to player evaluation
        dist = 0
        reward = -0.5
        for pellet in successorGameState.getCapsules():
            # get manhattan dist from pellet to newPos
            currDist = util.manhattanDistance(newPos, pellet)
            dist += currDist
            if currDist == 0:
                evaluationScore += 800

        evaluationScore += dist * reward



        # ghost to player evaluation
        dist = 0
        for ghost in newGhostStates:
            # get manhattan dist from ghost[0] to newPos
            currDist = util.manhattanDistance(newPos, ghost.getPosition())
            dist += currDist
            # accumulate w/ some reward
            if ghost.scaredTimer == 0:
                reward = .5
                # dist * reward
                evaluationScore += currDist * reward
                # -500 if pos = brave ghost pos
                if currDist == 0:
                    evaluationScore -= 1000
            # if scared, treat it like a capsule
            else:
                reward = -0.2
                # reward * dist
                evaluationScore += currDist * reward
                # reward if pos = scared ghost pos
                if currDist == 0:
                    evaluationScore += 800
                # reward for scared timer
                reward = 0.2
                evaluationScore += ghost.scaredTimer * reward

        if currFood[newPos[0]][newPos[1]]:
            evaluationScore += 10
        if len(newFood.asList()) == 0:
            evaluationScore += 1000

        return evaluationScore


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    # game.py -> Agent -> helper methods for "gameState"
    def getAction(self,gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # if state is terminal -> return the states utility
        # if gameState.isWin() or self.depth == 0:
        #     return gameState

        # we need to add cost for each depth that scales up the deeper

        #['Left', 'Right']
        # agentIndex=0 means Pacman, ghosts are >= 1

        # Pacman
        if self.index == 0:
            #---------MAXIMIZE----------

            v = -100000000000
            scores = []
            scores.append((v,""))

            # Pacman actions sucessors
            for action in gameState.getLegalActions(self.index):
                #print(gameState.generateSuccessor(self.index, action))
                pacmanActionSucessor = gameState.generateSuccessor(self.index, action)
                score = self.evaluationFunction(pacmanActionSucessor)
                #print("Pacman Successor Score: ", score)
                if score > v:
                    scores.append((score, action))
                else:
                    scores.append((v, action))

            #maximize pacman score
            maxPacman = max(scores)

            self.depth -= 1

            return maxPacman[1]

        # Ghost
        else:
            #--------MINIMIZE---------
            ghosts = []
            # for each ghost
            for i in range(gameState.getNumAgents()-1):
                v = 10000000000
                scores = []
                scores.append(v)
                #print(gameState.getLegalActions(i))
                # Ghost actions sucessors
                for action in gameState.getLegalActions(i):
                    #print(gameState.generateSuccessor(i, action))
                    ghostActionSucessor = gameState.generateSuccessor(i, action)
                    score = self.evaluationFunction(ghostActionSucessor)
                    #print("Ghost Successor Score: ", score)
                    if score < v:
                        scores.append((score, action))
                    else:
                        scores.append((v, action))
                
                #minimize ghosts score
                minGhost = min(scores)
                ghosts.append(minGhost)
            
            self.depth -= 1

            minGhosts = min(ghosts)

            return minGhosts[1]

    


        # Returns the total number of agents in the game
        print(gameState.getNumAgents())

        # Returns whether or not the game state is a winning state
        print(gameState.isWin())

        # Returns whether or not the game state is a losing state
        print(gameState.isLose())


        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
