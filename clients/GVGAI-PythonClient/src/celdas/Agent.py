import random

from AbstractPlayer import AbstractPlayer
from Types import *

from EpsilonStrategy import EpsilonStrategy
from ReplayMemory import ReplayMemory
from Experience import Experienceq

from utils.Types import LEARNING_SSO_TYPE
from utils.SerializableStateObservation import Observation
import math
import numpy as np
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tf_agents.environments import tf_py_environment

tf.compat.v1.enable_v2_behavior()

np.random.seed(91218)  # Set np seed for consistent results across runs
tf.set_random_seed(91218)

MEMORY_CAPACITY = 6
NUM_ACTIONS = 5
BATCH_SIZE = 1
GAMMA = 0.95
TAU = 0.08
state_size = 4

class Agent(AbstractPlayer):
    def __init__(self):
        AbstractPlayer.__init__(self)
        self.movementStrategy = EpsilonStrategy()
        self.replayMemory = ReplayMemory(MEMORY_CAPACITY)
        self.episode = 0

        self.policyNetwork = keras.Sequential([
            keras.layers.Dense(5, input_dim = 4, activation = 'relu'),
            keras.layers.Dense(NUM_ACTIONS, activation = 'softmax')
        ])
        self.targetNetwork = keras.Sequential([
            keras.layers.Dense(5, input_dim=4, activation='relu'),
            keras.layers.Dense(NUM_ACTIONS, activation='softmax')
        ])
        self.policyNetwork.compile(optimizer='adam',
                                   loss='mse',
                                   metrics=['accuracy'])
       
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
        self.episode += 1
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
        pprint(vars(sso))
        # print(self.get_perception(sso))
        currentPosition = self.getAvatarCoordinates(sso)

        if self.lastState is not None:
            reward = self.getReward(self.lastState, currentPosition)
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastActionIndex, sso, reward))
            # Train
            print('Train')
            loss = self.train(self.policyNetwork, self.replayMemory)
            print('Loss: ' + str(loss))

        
        action, index = self.getNextAction(sso)
        
        self.lastState = sso
        self.lastPosition = currentPosition
        if index is not None:
            self.lastActionIndex = index
        # print("Action and index: " + str(action) + " " + str(index))
        return action

    def train(self, policyNetwork, replayMemory, targetNetwork = None):
        if replayMemory.numSamples < BATCH_SIZE * 3:
            return 0
        batch = replayMemory.sample(BATCH_SIZE)
        asd = [np.ravel(self.get_perception(val.state)) for val in batch]
        print(asd)
        # bsd = [list(i) for i in asd]
        # print(bsd)
        states = tf_py_environment.TFPyEnvironment(np.array(asd))
        actions = np.array([val.actionIndex for val in batch])
        rewards = np.array([val.reward for val in batch])
        next_states = np.array([(np.zeros(state_size) if val.nextState is None else val.nextState) for val in batch])
        # predict Q(s,a) given the batch of states
        prim_qt = policyNetwork(states)
        # predict Q(s',a') from the evaluation network
        prim_qtp1 = policyNetwork(next_states)
        # copy the prim_qt into the target_q tensor - we then will update one index corresponding to the max action
        target_q = prim_qt.numpy()
        updates = rewards
        valid_idxs = np.array(next_states).sum(axis=1) != 0
        batch_idxs = np.arange(BATCH_SIZE)
        if targetNetwork is None:
            updates[valid_idxs] += GAMMA * \
                np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
        else:
            prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
            q_from_target = targetNetwork(next_states)
            updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs],
                                                                prim_action_tp1[valid_idxs]]
        target_q[batch_idxs, actions] = updates
        loss = policyNetwork.train_on_batch(states, target_q)
        if targetNetwork is not None:
            # update target network parameters slowly from primary network
            for t, e in zip(targetNetwork.trainable_variables, policyNetwork.trainable_variables):
                t.assign(t * (1 - TAU) + e * TAU)
        return loss

    def getNextAction(self, sso):
        # Do exploration or exploitation
        if self.movementStrategy.shouldExploit():
            #Exploitation
            index = self.replayMemory.sample(BATCH_SIZE).actionIndex
            action = sso.availableActions[index] 
            # print("Exploitation")
        else:
            #Exploration
            index = random.randint(0, len(sso.availableActions) - 1)
            action = sso.availableActions[index]
            # print("Exploration")
        return action, index

    def getAvatarCoordinates(self, state):
        position = state.avatarPosition
        return [int(position[0]/10), int(position[1]/10)]

    def getReward(self, lastState, currentPosition):
        level = self.get_perception(lastState)
        col = currentPosition[1] # col
        row = currentPosition[0] # row
        reward = 0
        if level[col][row] == 9 or level[col][row] == 3:
            # If we are in a safe spot or didn't move
            reward = -1
        elif level[col][row] == 2:
            # If we got the key
            reward = 100
        elif level[col][row] == 6:
            # If we are at the exit
            reward = 50
        elif level[col][row] == 5:
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
        sizeWorldHeightInPixels= sso.worldDimension[1]
        levelWidth = len(sso.observationGrid)
        levelHeight = len(sso.observationGrid[0])
        
        spriteSizeWidthInPixels =  sizeWorldWidthInPixels / levelWidth
        spriteSizeHeightInPixels =  sizeWorldHeightInPixels/ levelHeight
        level = np.array((levelHeight, levelWidth))
        level[:] = 9 # blank space
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
                return 0
            elif o.itype == 0:
                return 1 # Wall
            elif o.itype == 4:
                return 2 # Key
            else:
                return 3 # Agent
            
             
        elif o.category == 0:
            if o.itype == 5:
                return 3
            elif o.itype == 6:
                return 4
            elif o.itype == 1:
                return 3
            else:
                return 3
             
        elif o.category == 6:
            return 5 # Enemy
        elif o.category == 2:
            return 6 # Exit
        elif o.category == 3:
            if o.itype == 1:
                return 5
            else:         
                return 5         
        elif o.category == 5:
            if o.itype == 5:
                return 7
            else:         
                return 5
        else:                          
            return 8

