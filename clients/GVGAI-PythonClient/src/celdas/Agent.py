import random

from AbstractPlayer import AbstractPlayer
from Types import *
from EpsilonStrategy import EpsilonStrategy

from utils.Types import LEARNING_SSO_TYPE
from utils.SerializableStateObservation import Observation
import math
import numpy as np
from pprint import pprint

class Agent(AbstractPlayer):
    def __init__(self):
        AbstractPlayer.__init__(self)

    movementStrategy = EpsilonStrategy()
       
    """
    * Public method to be called at the start of every level of a game.
    * Perform any level-entry initialization here.
    * @param sso Phase Observation of the current game.
    * @param elapsedTimer Timer (1s)
    """

    def init(self, sso, elapsedTimer):
        self.lastState = None
        self.lastPosition = None
        self.lastActionIndex = None
        print("Game initialized")

    """
     * Method used to determine the next move to be performed by the agent.
     * This method can be used to identify the current state of the game and all
     * relevant details, then to choose the desired course of action.
     *
     * @param sso Observation of the current state of the game to be used in deciding
     *            the next action to be taken by the agent.
     * @param elapsedTimer Timer (40ms)
     * @return The action to be performed by the agent.
     """

    def act(self, sso, elapsedTimer):        
        # pprint(vars(sso))
        # print(self.get_perception(sso))
        currentPosition = self.getAvatarCoordinates(sso)

        if self.lastState is not None:
            reward = self.getReward(self.lastState, currentPosition)
            # print("Reward: " + str(reward))
        
        action, index = self.getNextAction(sso)
        
        self.lastState = sso
        self.lastPosition = currentPosition
        if index is not None:
            self.lastActionIndex = index
        # print("Action and index: " + str(action) + " " + str(index))
        return action
    
    def getNextAction(self, sso):
        # Do exploration or exploitation
        if self.movementStrategy.shouldExploit():
            #Exploitation
            # TODO: get from memory
            index = random.randint(0, len(sso.availableActions) - 1)
            action = sso.availableActions[index] 
            print("Exploitation")
        else:
            #Exploration
            index = random.randint(0, len(sso.availableActions) - 1)
            action = sso.availableActions[index]
            print("Exploration")
        return action, index

    def getAvatarCoordinates(self, state):
        position = state.avatarPosition
        return [int(position[0]/10), int(position[1]/10)]

    def getReward(self, lastState, currentPosition):
        level = self.get_perception(lastState)
        x = currentPosition[1] # col
        y = currentPosition[0] # row
        reward = 0
        if level[x][y] == "." or level[x][y] == "A":
            # If we are in a safe spot or didn't move
            reward = -1
        elif level[x][y] == "L":
            # If we got the key
            reward = 50
        elif level[x][y] == "S":
            # If we are at the exit
            reward = 100
        elif level[x][y] == "e":
            # If we touched an enemy
            reward = -100
        return reward

    """
    * Method used to perform actions in case of a game end.
    * This is the last thing called when a level is played (the game is already in a terminal state).
    * Use this for actions such as teardown or process data.
    *
    * @param sso The current state observation of the game.
    * @param elapsedTimer Timer (up to CompetitionParameters.TOTAL_LEARNING_TIME
    * or CompetitionParameters.EXTRA_LEARNING_TIME if current global time is beyond TOTAL_LEARNING_TIME)
    * @return The next level of the current game to be played.
    * The level is bound in the range of [0,2]. If the input is any different, then the level
    * chosen will be ignored, and the game will play a random one instead.
    """

    def result(self, sso, elapsedTimer):
        return random.randint(0, 2)

    def get_perception(self, sso):
        sizeWorldWidthInPixels= sso.worldDimension[0]
        sizeWorldHeightInPixels= sso.worldDimension[1];
        levelWidth = len(sso.observationGrid);
        levelHeight = len(sso.observationGrid[0]);
        
        spriteSizeWidthInPixels =  sizeWorldWidthInPixels / levelWidth;
        spriteSizeHeightInPixels =  sizeWorldHeightInPixels/ levelHeight;
        level = np.chararray((levelHeight, levelWidth))
        level[:] = '.'
        avatar_observation = Observation()
        for ii in range(levelWidth):                    
            for jj in range(levelHeight):
                listObservation = sso.observationGrid[ii][jj]
                if len(listObservation) != 0:
                    aux = listObservation[len(listObservation)-1]
                    if aux is None: continue
                    level[jj][ii] = self.detectElement(aux)
    

        return level

    def detectElement(self, o):
        if o.category == 4:
            if o.itype == 3:
                return '0'
            elif o.itype == 0:
                return 'w' # Wall
            elif o.itype == 4:
                return 'L' # Key
            else:
                return 'A' # Agent
            
             
        elif o.category == 0:
            if o.itype == 5:
                return 'A'
            elif o.itype == 6:
                return 'B'
            elif o.itype == 1:
                return 'A'
            else:
                return 'A'
             
        elif o.category == 6:
            return 'e' # Enemy
        elif o.category == 2:
            return 'S' # Exit
        elif o.category == 3:
            if o.itype == 1:
                return 'e'
            else:         
                return 'e'         
        elif o.category == 5:
            if o.itype == 5:
                return 'x'
            else:         
                return 'e'
        else:                          
            return '?'

