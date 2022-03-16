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
import search
import searchAgents

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
        # print("Sucessor New Postion: ", newPos)

        # matrix of T/F
        currFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        # print("Sucessor New Food: ", newFood)

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

    def maxValue(self, depth, state):

        # check if terminal or max depth achieved
        if self.depth == depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        # get actions of pacman
        actions = state.getLegalActions(0)

        # get successors of pacman based on actions
        successors = [state.generateSuccessor(0, action) for action in actions]

        # get scores of ghosts recursively
        scores = [self.minValue(depth, state, 1) for state in successors]

        # maximize pac man score
        return max(scores)

    def minValue(self, depth, state, index):

        # check if terminal or max depth achieved
        if self.depth == depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        # get actions of ghosts
        actions = state.getLegalActions(index)

        # get successors of ghost based on actions
        successors = [state.generateSuccessor(index, action) for action in actions]

        # check number of adversaries
        # print(state.getNumAgents()-1);
        if index >= state.getNumAgents() - 1:
            # pacman - increase depth by 1 - maximize
            scores = [self.maxValue(depth + 1, state) for state in successors]
        else:
            # ghost - minimize
            scores = [self.minValue(depth, state, index + 1) for state in successors]

        # minimize ghosts
        return min(scores)

    # game.py -> Agent -> helper methods for "gameState"
    def getAction(self, gameState):
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

        # get pacman actions
        actions = gameState.getLegalActions(0)

        # get successors of pacman for each action
        successors = [gameState.generateSuccessor(0, action) for action in actions]

        # get scores from minimizer - recursively visits each layer
        scores = [self.minValue(0, state, 1) for state in successors]

        # maximize pac man
        bestScore = max(scores)

        # choose best score for pacman
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        # return action
        return actions[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):

    INF = 2147483647
    NEG_INF = -2147483648

    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxValue(self, depth, state, alpha, beta):

        v = self.NEG_INF

        # check if terminal or max depth achieved
        if self.depth == depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        # get actions of pacman
        actions = state.getLegalActions(0)

        for action in actions:
            # get successor of pacman based on action
            nextState = state.generateSuccessor(0, action)
            # get scores of ghosts recursively and maximize score
            v = max(v, self.minValue(depth, nextState, 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)

        return v

    def minValue(self, depth, state, index, alpha, beta):

        v = self.INF

        # check if terminal or max depth achieved
        if self.depth == depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        # get actions of ghosts
        actions = state.getLegalActions(index)

        for action in actions:
            # get successor of ghost based on action
            newState = state.generateSuccessor(index, action)
            if index >= state.getNumAgents() - 1:
                # pacman - maximize
                score = self.maxValue(depth + 1, newState, alpha, beta)
            else:
                # ghost - minimize
                score = self.minValue(depth, newState, index + 1, alpha, beta)
            # minimize return value
            v = min(v, score)
            if v < alpha:
                return v
            beta = min(beta, v)

        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # get pacman actions
        actions = gameState.getLegalActions(0)

        v = self.NEG_INF
        scores = []
        alpha = self.NEG_INF
        beta = self.INF

        for action in actions:
            # get successor of pacman for each action
            state = gameState.generateSuccessor(0, action)
            # get score from recursive minimizer
            score = self.minValue(0, state, 1, alpha, beta)
            v = max(v, score)
            scores.append(score)
            # no need to prune because beta never changes at root
            alpha = max(alpha, v)

        # maximize pac man
        bestScore = max(scores)

        # choose best score for pacman
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        # return action
        return actions[chosenIndex]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxValue(self, depth, state):

        # check if terminal or max depth achieved
        if self.depth == depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        # get actions of pacman
        actions = state.getLegalActions(0)

        # get successors of pacman based on actions
        successors = [state.generateSuccessor(0, action) for action in actions]

        # get scores of ghosts recursively
        scores = [self.expValue(depth, state, 1) for state in successors]

        # maximize pac man score
        return max(scores)

    def expValue(self, depth, state, index):
        # in minValue, find expected value instead of min
        # P(successor) = 1 / # of successors

        # check if terminal or max depth achieved
        if self.depth == depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        # get actions of ghosts
        actions = state.getLegalActions(index)

        # get successors of ghost based on actions
        successors = [state.generateSuccessor(index, action) for action in actions]

        # check number of adversaries
        # print(state.getNumAgents()-1);
        if index >= state.getNumAgents() - 1:
            # pacman - increase depth by 1 - maximize
            scores = [self.maxValue(depth + 1, state) * (1 / len(successors)) for state in successors]
        else:
            # ghost - minimize
            scores = [self.expValue(depth, state, index + 1) * (1 / len(successors)) for state in successors]

        # minimize ghosts
        return sum(scores)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # get pacman actions
        actions = gameState.getLegalActions(0)

        # get successors of pacman for each action
        successors = [gameState.generateSuccessor(0, action) for action in actions]

        # get scores from minimizer - recursively visits each layer
        scores = [self.expValue(0, state, 1) for state in successors]

        # maximize pac man
        bestScore = max(scores)

        # choose best score for pacman
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        # return action
        return actions[chosenIndex]

def depthFirstSearch(currentGameState, depth):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    "*** YOUR CODE HERE ***"

    # Strategy: expand a deepest node first
    # Frontier is a stack in util.py

    closed = set()
    fringe = util.Stack()

    if len(currentGameState.getLegalActions()) == 0:
        return None

    # add start state with an empty action list
    fringe.push((currentGameState, []))
    while True:
        if fringe.isEmpty():
            # TODO: use a better error
            util.raiseNotDefined()
        node = fringe.pop()
        currState = node[0]
        currActions = node[1]
        if self.currentGameState.isLose() or currentGameState.isWin():
            return currActions
        if currState not in closed:
            closed.add(currState)
            # for (state, action, cost) in currentGameState.getSuccessors(currState):
            for action in currentGameState.getLegalActions():
                successor = currentGameState.generatePacmanSuccessor(action)
                # keep track of all actions leading to this one
                newActions = currActions + [action]
                fringe.push((successor, newActions))

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    #prob = searchAgents.PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(depthFirstSearch(gameState, gameState.depth))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Here are some useful elements of the startState
    walls = currentGameState.getWalls()
    foods = currentGameState.getFood().asList()
    sum = 0
    currentPacmanPosition = currentGameState.getPacmanPosition()
    for food in foods:
        sum += mazeDistance(currentPacmanPosition, food, currentGameState)

    sum = sum/len(foods)

    #get adversaries
    ghosts = currentGameState.getGhostStates()

    for ghost in ghosts:
        if ghost.scaredTimer > 0:
            # if scared give positive cost the closer they are
            sum += mazeDistance(currentPacmanPosition, ghost.getPosition(), currentGameState)
        else:
            # else give negative cost the closer they are
            sum -= mazeDistance(currentPacmanPosition, ghost.getPosition(), currentGameState)

    sum = sum / len(ghosts)

    return sum

# Abbreviation
better = betterEvaluationFunction
