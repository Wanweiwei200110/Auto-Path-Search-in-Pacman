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


from util import manhattanDistance
from game import Directions
import random, util

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return float('inf')
        if action == Directions.STOP:
            return float('-inf')
        score = successorGameState.getScore()

        walls = currentGameState.getWalls()
        top, right = walls.height - 2, walls.width - 2
        if newPos[0] <= 2 or newPos[0] >= right - 1 or newPos[1] <= 2 or newPos[1] >= top - 1:
            score -= 2

        num_agents = currentGameState.getNumAgents()
        for ind in range(1, num_agents):
            next_ghost_pos = successorGameState.getGhostPosition(ind)
            dist_to_ghost = manhattanDistance(newPos, next_ghost_pos)
            if newScaredTimes[ind-1] < 2:
                if dist_to_ghost <= 1:
                    return float('-inf')
                else:
                    score += 1 / (dist_to_ghost + 0.1)
            else:
                if dist_to_ghost <= 1:
                    score += 200

        foods = newFood.asList()
        for food in foods:
            dist_to_food = manhattanDistance(newPos, food)
            if dist_to_food == 0:
                score += 200
            else:
                score += 4 ** (-1 * dist_to_food) * 20

        capsules = currentGameState.getCapsules()
        if newPos in capsules:
            score += 300

        # score += len(currentGameState.getLegalPacmanActions()) / 5
        # score += random.random()
        return score


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def df_minimax(self, state, depth, num_agents):
        best_move = 'Stop'
        if depth == 0 or state.isWin() or state.isLose():
            return best_move, self.evaluationFunction(state)
        if depth % num_agents == 0:
            turn = 0
            value = float('-inf')
        else:
            turn = num_agents - depth % num_agents
            value = float('inf')
        actions = state.getLegalActions(turn)
        for move in actions:
            next_state = state.generateSuccessor(turn, move)
            next_move, next_val = self.df_minimax(next_state, depth-1, num_agents)
            if turn == 0 and value < next_val:
                value, best_move = next_val, move
            if turn and value > next_val:
                value, best_move = next_val, move
        return best_move, value

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
        depth = self.depth
        num_agents = gameState.getNumAgents()
        return self.df_minimax(gameState, depth*num_agents, num_agents)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, state, depth, num_agents, alpha, beta):
        best_move = 'Stop'
        if depth == 0 or state.isWin() or state.isLose():
            return best_move, self.evaluationFunction(state)
        if depth % num_agents == 0:
            turn = 0
            value = float('-inf')
        else:
            turn = num_agents - depth % num_agents
            value = float('inf')
        actions = state.getLegalActions(turn)
        for move in actions:
            next_state = state.generateSuccessor(turn, move)
            next_move, next_val = self.alphabeta(next_state, depth-1, num_agents, alpha, beta)
            if turn == 0:
                if value < next_val:
                    value, best_move = next_val, move
                if value >= beta:
                    return best_move, value
                alpha = max(alpha, value)
            else:
                if value > next_val:
                    value, best_move = next_val, move
                if value <= alpha:
                    return best_move, value
                beta = min(beta, value)
        return best_move, value

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        num_agents = gameState.getNumAgents()
        return self.alphabeta(gameState, depth*num_agents, num_agents, float('-inf'), float('inf'))[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expecimax(self, state, depth, num_agents):
        best_move = 'Stop'
        if depth == 0 or state.isWin() or state.isLose():
            return best_move, self.evaluationFunction(state)
        if depth % num_agents == 0:
            turn = 0
            value = float('-inf')
        else:
            turn = num_agents - depth % num_agents
            value = 0
        actions = state.getLegalActions(turn)
        for move in actions:
            next_state = state.generateSuccessor(turn, move)
            next_move, next_val = self.expecimax(next_state, depth-1, num_agents)
            if turn == 0 and value < next_val:
                value, best_move = next_val, move
            if turn:
                value += float(next_val) / len(actions)
        return best_move, value

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        num_agents = gameState.getNumAgents()
        return self.expecimax(gameState, depth*num_agents, num_agents)[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    The score is calculated by the base score(current score) and a linear
    combination of the following features:
    - number of foods left
    - number of capsules left
    - min dist to food
    - min dist to capsule
    - min dist to regular ghost
    - min dist to scared ghost
    Where the negatively weighted features give the pacman incentive to get closer,
    and the positively weighted features give the pacman incentive to stay away.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')

    score = currentGameState.getScore()
    Pos = currentGameState.getPacmanPosition()

    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    dist_to_scared_ghost = []
    dist_to_regular_ghost = []
    num_agents = currentGameState.getNumAgents()
    for ind in range(1, num_agents):
        ghost_pos = currentGameState.getGhostPosition(ind)
        dist = manhattanDistance(Pos, ghost_pos)
        if newScaredTimes[ind-1] < 2:
            dist_to_regular_ghost.append(dist)
        else:
            dist_to_scared_ghost.append(dist)
    min_dist_to_regular_ghost = min(dist_to_regular_ghost) if dist_to_regular_ghost else 0
    min_dist_to_scared_ghost = min(dist_to_scared_ghost) if dist_to_scared_ghost else 0

    dist_to_food = []
    foods = currentGameState.getFood().asList()
    for food in foods:
        dist_to_food.append(manhattanDistance(Pos, food))
    min_dist_to_food = min(dist_to_food) if dist_to_food else 0

    dist_to_capsule = []
    capsules = currentGameState.getCapsules()
    for capsule in capsules:
        dist_to_capsule.append(util.manhattanDistance(Pos, capsule))
    min_dist_to_capsule = min(dist_to_capsule) if dist_to_capsule else 0

    score = score - 25 * len(foods) - 100 * len(capsules) - 3 * min_dist_to_capsule - 1 * min_dist_to_food\
            - 0.4 * min_dist_to_regular_ghost - 5 * min_dist_to_scared_ghost

    return score

# Abbreviation
better = betterEvaluationFunction
