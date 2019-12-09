import random
import os
import datetime as dt

from AbstractPlayer import AbstractPlayer
from Types import *

from EpsilonStrategy import EpsilonStrategy
from ReplayMemory import ReplayMemory
from Experience import Experience

from utils.Types import LEARNING_SSO_TYPE
from utils.SerializableStateObservation import Observation
import math
import numpy as np
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from scipy.spatial import distance

tf.compat.v1.enable_v2_behavior()

# np.random.seed(91218)  # Set np seed for consistent results across runs
# tf.set_random_seed(91218)

MEMORY_CAPACITY = 5000
NUM_ACTIONS = 5
BATCH_SIZE = 6
GAMMA = 0.95
TAU = 0.08
state_size = 4
STORE_PATH = os.getcwd()
train_writer = tf.summary.create_file_writer(
    STORE_PATH + "/logs/Zelda_{}".format(dt.datetime.now().strftime('%d%m%Y%H%M')))

actionToFloat = {'ACTION_NIL': 0.0,
       'ACTION_UP': 1.0,
       'ACTION_LEFT': 2.0,
       'ACTION_DOWN': 3.0,
       'ACTION_RIGHT': 4.0,
       'ACTION_USE': 5.0,
       'ACTION_ESCAPE': 6.0}


class Agent(AbstractPlayer):
    def __init__(self):
        AbstractPlayer.__init__(self)
        self.movementStrategy = EpsilonStrategy()
        self.replayMemory = ReplayMemory(MEMORY_CAPACITY)
        self.episode = 0

        networkOptions = [
            keras.layers.Dense(100, input_dim=122, activation='relu'),
            # keras.layers.Dense(
            #     200, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(
                200, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(
            #     150, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(NUM_ACTIONS)
        ]

        self.policyNetwork = keras.Sequential(networkOptions)
        self.targetNetwork = keras.Sequential(networkOptions)
        self.policyNetwork.compile(optimizer='adam',
                                   loss='mse')
        print(self.policyNetwork.summary())
       
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
        self.averageLoss = 0
        self.gameOver = False
        self.cnt = 0
        self.steps = 0
        self.gotTheKey = False
        self.keyPosition = None
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
        # pprint(sso.NPCPositions)
        # print(self.get_perception(sso))
        currentPosition = self.getAvatarCoordinates(sso)
        if not self.gotTheKey:
            try:
                self.keyPosition = sso.immovablePositions[1][0].getPositionAsArray()
            except:
                self.gotTheKey = True

        if self.lastState is not None:
            reward = self.getReward(self.lastState, currentPosition, sso)
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastActionIndex, reward, sso))
            # Train
            loss = self.train(self.policyNetwork, self.replayMemory, self.targetNetwork)
            self.averageLoss += loss
            self.cnt += reward

        index = self.getNextAction(sso, self.policyNetwork)
        action = sso.availableActions[index]
        self.lastState = sso
        self.lastPosition = currentPosition
        if index is not None:
            self.lastActionIndex = index
        # print("Action and index: " + str(action) + " " + str(index))
        self.steps += 1
        return action

    # def stateToTensor(self, state):
        # return tf.convert_to_tensor([np.ravel(self.get_perception(state))], dtype=tf.float32)


    # Modify here to alter network inputs, be careful of dynamic arrays and to change network inputs
    def buildNetworkInput(self, state):
        perception = np.ravel(self.get_perception(state))
        # perception = np.append(perception, state.gameScore)
        # perception = np.append(perception, 0.0 if state.isGameOver else 1.0)
        perception = np.append(perception, 0.0 if not self.gotTheKey else 1.0)
        perception = np.append(perception, actionToFloat[state.avatarLastAction])
        perception = np.append(perception, np.ravel(state.avatarOrientation))
        # perception = np.append(perception, len(state.NPCPositions)) # number of enemies
        # perception = np.append(perception, np.ravel([i.getPositionAsArray() for i in np.ravel(state.portalsPositions)]))
        # perception = np.append(perception, np.ravel(
            # [i.getPositionAsArray() for i in np.ravel(state.NPCPositions)]))
        perception = np.append(perception, self.getDistanceToKey(state))
        # perception = np.append(perception, np.ravel(
        #     [i.getPositionAsArray() for i in np.ravel(state.resourcesPositions)]))
        return perception

    def train(self, policyNetwork, replayMemory, targetNetwork = None):
        if replayMemory.numSamples < BATCH_SIZE * 3:
            return 0
        batch = replayMemory.sample(BATCH_SIZE)
        # rawStates = [np.ravel(self.get_perception(val.state)) for val in batch]
        rawStates = [self.buildNetworkInput(val.state) for val in batch]
        states = tf.convert_to_tensor(rawStates, dtype=tf.float32)
        actions = np.array([val.actionIndex for val in batch])
        rewards = np.array([val.reward for val in batch])
        rawNextStates = [(np.zeros(state_size) if val.nextState is None else val.nextState) for val in batch]
        # preTensorNextStates = [np.ravel(self.get_perception(b)) for b in rawNextStates]
        preTensorNextStates = [self.buildNetworkInput(val.state) for vcal in rawNextStates]
        nextStates = tf.convert_to_tensor(preTensorNextStates, dtype=tf.float32)
        # predict Q(s,a) given the batch of states
        prim_qt = policyNetwork(states)
        # predict Q(s',a') from the evaluation network
        prim_qtp1 = policyNetwork(nextStates)
        # copy the prim_qt into the target_q tensor - we then will update one index corresponding to the max action
        target_q = prim_qt.numpy()
        updates = rewards
        valid_idxs = np.array(nextStates).sum(axis=1) != 0
        batch_idxs = np.arange(BATCH_SIZE)
        if targetNetwork is None:
            updates[valid_idxs] += GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
        else:
            prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
            q_from_target = targetNetwork(nextStates)
            updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs],
                                                                prim_action_tp1[valid_idxs]]
        target_q[batch_idxs, actions] = updates
        loss = policyNetwork.train_on_batch(states, target_q)
        if targetNetwork is not None:
            # update target network parameters slowly from primary network
            for t, e in zip(targetNetwork.trainable_variables, policyNetwork.trainable_variables):
                t.assign(t * (1 - TAU) + e * TAU)
        return loss

    def getNextAction(self, state, policyNetwork):
        # Do exploration or exploitation
        if self.movementStrategy.shouldExploit():
            #Exploitation
            # print('Exploitation')
            sd = tf.reshape(policyNetwork(tf.convert_to_tensor(
                [self.buildNetworkInput(state)], dtype=tf.float32)), (1, -1))
            return np.argmax(sd)
        else:
            #Exploration
            # print('Exploration')
            return random.randint(0, NUM_ACTIONS - 1)

    def getAvatarCoordinates(self, state):
        position = state.avatarPosition
        return [float(position[1]/10), float(position[0]/10)]

    def getDistanceToKey(self, state):
        distToKey = distance.cityblock(
            self.getAvatarCoordinates(state), self.keyPosition)
        return 0.0 if self.gotTheKey else distToKey

    def isCloserToKey(self, lastState, currentState):
        return self.getDistanceToKey(currentState) < self.getDistanceToKey(lastState)

    def getReward(self, lastState, currentPosition, currentState):
        level = self.get_perception(lastState)
        col = int(currentPosition[0]) # col
        row = int(currentPosition[1]) # row
        reward = 0.0
        if currentState.NPCPositions and len(currentState.NPCPositions[0]) < len(lastState.NPCPositions[0]):
            print('KILLED AN ENEMY')
            reward = 10.0
        # elif self.keyPosition is not None and np.linalg.norm(np.array(lastState.avatarPosition)-np.array(self.keyPosition)) < np.linalg.norm(np.array(currentState.avatarPosition)-np.array(self.keyPosition)):
        #     print('closer')
        #     reward = 5.0
        elif self.keyPosition is not None and self.isCloserToKey(lastState, currentState):
            reward = 10.0
        elif level[col][row] == 2:
            # If we got the key
            print('GOT THE KEY')
            self.gotTheKey = True
            reward = 1000.0
        elif level[col][row] == 6 and self.gotTheKey:
            # If we are at the exit
            print('WON')
            reward = 2000.0
        elif level[col][row] == 5:
            # If we touched an enemy
            reward = -50.0
        elif level[col][row] == 9 or level[col][row] == 3:
            # If we are in a safe spot or didn't move
            reward = -5.0
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
        self.gameOver = True
        if self.lastActionIndex is not None:
            reward = self.getReward(self.lastState, self.getAvatarCoordinates(sso), sso)
            if not sso.isAvatarAlive:
                reward = -1000.0
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastActionIndex, reward, sso))
        self.episode += 1

        if self.gameOver:
            self.averageLoss /= self.cnt
            print("Episode: {}, Reward: {}, avg loss: {}, eps: {}".format(
                self.episode, self.cnt, self.averageLoss, self.movementStrategy.epsilon))
            with train_writer.as_default():
                tf.summary.scalar('reward', self.cnt, step=self.steps)
                tf.summary.scalar(
                    'avg loss', self.averageLoss, step=self.steps)
        return random.randint(0, 2)

    def get_perception(self, sso):
        sizeWorldWidthInPixels= sso.worldDimension[0]
        sizeWorldHeightInPixels= sso.worldDimension[1]
        levelWidth = len(sso.observationGrid)
        levelHeight = len(sso.observationGrid[0])
        
        spriteSizeWidthInPixels =  sizeWorldWidthInPixels / levelWidth
        spriteSizeHeightInPixels =  sizeWorldHeightInPixels/ levelHeight
        level = np.zeros((levelHeight, levelWidth))
        level[:] = 9.0 # blank space
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
                return 0.0
            elif o.itype == 0:
                return 1.0 # Wall
            elif o.itype == 4:
                return 2.0 # Key
            else:
                return 3.0 # Agent
            
             
        elif o.category == 0:
            if o.itype == 5:
                return 3.0
            elif o.itype == 6:
                return 4.0
            elif o.itype == 1:
                return 3.0
            else:
                return 3.0
             
        elif o.category == 6:
            return 5.0 # Enemy
        elif o.category == 2:
            return 6.0 # Exit
        elif o.category == 3:
            if o.itype == 1:
                return 5.0
            else:         
                return 5.0      
        elif o.category == 5:
            if o.itype == 5:
                return 7.0
            else:         
                return 5.0
        else:                          
            return 8.0

