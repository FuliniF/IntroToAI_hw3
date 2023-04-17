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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        # raise NotImplementedError("To be implemented")

        # initial agent  = 0 (pacman) => find max value first
        # initial cDepth = 0 (current depth)
        score, Action = self.max_value(0, 0, gameState)
        return Action

    def max_value(self, agentIndex, cDepth, gameState): # agentIndex = 0(pacman)
        if gameState.isWin() or gameState.isLose() or cDepth == self.depth:
            return self.evaluationFunction(gameState), None

        legalActions = gameState.getLegalActions(agentIndex)
        bestV = -10000
        bestMove = "x"
        for action in legalActions:
            # check every value possible after current state
            score, _ = self.min_value(1, cDepth, gameState.getNextState(0, action))
            if bestV < score:
                bestV = score
                bestMove = action

        return bestV, bestMove

    def min_value(self, agentIndex, cDepth, gameState): # agentIndex != 0(ghost)
        if gameState.isWin() or gameState.isLose() or cDepth == self.depth:
            return self.evaluationFunction(gameState), None

        legalActions = gameState.getLegalActions(agentIndex)
        bestV = 10000
        bestMove = "x"
        if gameState.getNumAgents() == (agentIndex+1):
            nextAgent = 0
            cDepth += 1 # last ghost reached, current depth++
        else:
            nextAgent = agentIndex + 1
        for action in legalActions:
            # check every value possible after current state
            if nextAgent == 0: # the last ghost
                score, _ = self.max_value(0, cDepth, gameState.getNextState(agentIndex, action))
            else:
                score, _ = self.min_value(nextAgent, cDepth, gameState.getNextState(agentIndex, action))
            if bestV > score:
                bestV = score
                bestMove = action

        return bestV, bestMove
            
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        # raise NotImplementedError("To be implemented")

        # initial agent  = 0 (pacman) => find max value first
        # initial cDepth = 0 (current depth)
        # initial alpha  = -10000, beta = 10000
        score, Action = self.max_value(0, 0, gameState, -10000, 10000)
        return Action

    def max_value(self, agentIndex, cDepth, gameState, alpha, beta): # agentIndex = 0(pacman)
        if gameState.isWin() or gameState.isLose() or cDepth == self.depth:
            return self.evaluationFunction(gameState), None

        legalActions = gameState.getLegalActions(agentIndex)
        bestV = -10000
        bestMove = "x"
        for action in legalActions:
            score, _ = self.min_value(1, cDepth, gameState.getNextState(0, action), alpha, beta)
            if bestV < score:
                bestV = score
                bestMove = action

            # prune
            if score > beta:
                break
            if score > alpha:
                alpha = score

        return bestV, bestMove

    def min_value(self, agentIndex, cDepth, gameState, alpha, beta): # agentIndex != 0(ghost)
        if gameState.isWin() or gameState.isLose() or cDepth == self.depth:
            return self.evaluationFunction(gameState), None

        legalActions = gameState.getLegalActions(agentIndex)
        bestV = 10000
        bestMove = "x"
        if gameState.getNumAgents() == (agentIndex+1):
            nextAgent = 0
            cDepth += 1 # last ghost reached, current depth++
        else:
            nextAgent = agentIndex + 1
        for action in legalActions:
            if nextAgent == 0:
                score, _ = self.max_value(0, cDepth, gameState.getNextState(agentIndex, action), alpha, beta)
            else:
                score, _ = self.min_value(nextAgent, cDepth, gameState.getNextState(agentIndex, action), alpha, beta)
            if bestV > score:
                bestV = score
                bestMove = action
            
            # prune
            if score < alpha:
                break
            if score < beta:
                beta = score

        return bestV, bestMove

        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        # raise NotImplementedError("To be implemented")

        # initial agent  = 0 (pacman) => find max value first
        # initial cDepth = 0 (current depth)
        score, Action = self.max_value(0, 0, gameState)
        return Action

    #  this part is same as minimax
    def max_value(self, agentIndex, cDepth, gameState): # agentIndex = 0(pacman)
        if gameState.isWin() or gameState.isLose() or cDepth == self.depth:
            return self.evaluationFunction(gameState), None
        legalActions = gameState.getLegalActions(agentIndex)
        bestV = -10000
        bestMove = "x"
        for action in legalActions:
            score = self.exp_value(1, cDepth, gameState.getNextState(0, action))
            if bestV < score:
                bestV = score
                bestMove = action

        return bestV, bestMove

    # this part has changed into calculating expected value
    def exp_value(self, agentIndex, cDepth, gameState): # agentIndex != 0(ghost)
        if gameState.isWin() or gameState.isLose() or cDepth == self.depth:
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(agentIndex)

        # expect_score : add all score first, return (all_score/num_of_actions) in the end
        expect_score = 0
        if gameState.getNumAgents() == (agentIndex+1):
            nextAgent = 0
            cDepth += 1
        else:
            nextAgent = agentIndex + 1
        for action in legalActions:
            if nextAgent == 0:
                score, _ = self.max_value(0, cDepth, gameState.getNextState(agentIndex, action))
                expect_score += score
                
            else:
                score = self.exp_value(nextAgent, cDepth, gameState.getNextState(agentIndex, action))
                expect_score += score

        return expect_score / len(legalActions)
            
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")

    # get infomation first
    PacPos = currentGameState.getPacmanPosition()
    legalActions = currentGameState.getLegalActions(0)

    # food info
    if currentGameState.getFood().asList():
        nearestFoodDistance = min([manhattanDistance(PacPos, food) for food in currentGameState.getFood().asList()])
    else:
        nearestFoodDistance = 0

    # capsule info
    capsules = currentGameState.getCapsules()
    nearestCapsuleDistance = 0
    if len(capsules) != 0:
        capsulesDistance = [manhattanDistance(PacPos, capsule) for capsule in capsules]
        nearestCapsuleDistance = min(capsulesDistance)

    # ghost info
    GhostStates = currentGameState.getGhostStates()
    minGhostDistance = 1000
    for ghostState in GhostStates:
        dist = manhattanDistance(PacPos, ghostState.getPosition())
        if dist < minGhostDistance:
            minGhostDistance = dist
            nearestScaredTime = ghostState.scaredTimer
    isScared = nearestScaredTime > 1
    
    # initialize Score num
    Score = currentGameState.getScore() * 10

    """
    rules:
    1. if near food then go to eat
    2. if near capsule then go to eat
    3. if near ghost and ghost is scared then go to eat ghost
    4. if near ghost but not scared then run away
    5. STOP is the least preferrable action
    """

    # 1.
    Score -= 10 * nearestFoodDistance

    # 2.
    if nearestCapsuleDistance <= 5:
        Score -= 20 * nearestCapsuleDistance

    # 3.
    if isScared:
        Score -= 2 * minGhostDistance
    # 4.
    else:
        Score += 10 * minGhostDistance

    # 5.
    for action in legalActions:
        if action == Directions.STOP:
            Score += 1000

    return Score
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
