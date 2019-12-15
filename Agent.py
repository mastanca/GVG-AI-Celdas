from EpsilonStrategy import EpsilonStrategy
from ReplayMemory import ReplayMemory
from Experience import Experience

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import datetime as dt
import os
from scipy.spatial import distance

MEMORY_CAPACITY = 50000  # @param {type:"integer"}
state_size = 117  # @param {type:"integer"}
NUM_ACTIONS = 5  # @param {type:"integer"}
ALPHA = 0.001  # @param {type:"integer"}
batch_size = 32  # @param {type:"integer"}
GAMMA = 0.95  # @param {type:"integer"}
TAU = 0.08  # @param {type:"integer"}

STORE_PATH = os.getcwd()
train_writer = tf.summary.create_file_writer(
    STORE_PATH + "/logs/Zelda_{}".format(dt.datetime.now().strftime('%d%m%Y%H%M')))

class Agent():
    def __init__(self):
        self.movementStrategy = EpsilonStrategy()
        self.replayMemory = ReplayMemory(MEMORY_CAPACITY)
        self.episode = 0

        networkOptions = [
            keras.layers.Dense(
                state_size, input_dim=state_size, activation='relu'),
            keras.layers.Dense(
                100, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(
                100, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(NUM_ACTIONS)
        ]

        self.policyNetwork = keras.Sequential(networkOptions)
        self.targetNetwork = keras.Sequential(networkOptions)
        self.policyNetwork.compile(optimizer=keras.optimizers.Adam(learning_rate=ALPHA),
                                   loss=keras.losses.mean_squared_error)
        print(self.policyNetwork.summary())
        try:
            self.policyNetwork.load_weights("./network/zelda-ddqn.h5")
            self.movementStrategy.epsilon = 0.01
            print('Model loaded')
        except:
            print('Model file not found')

    """
    * Public method to be called at the start of every level of a game.
    * Perform any level-entry initialization here.
    """

    def init(self):
        self.lastState = None
        self.lastPosition = None
        self.lastActionIndex = None
        self.averageLoss = 0
        self.averageReward = 0
        self.gameOver = False
        self.cnt = 0
        self.steps = 0
        self.gotTheKey = False
        self.keyPosition = None
        self.closerToExit = False
        self.closerToKey = False
        print("Game initialized")

    def act(self, state):
        # pprint(vars(state))
        # pprint(state.NPCPositions)
        # print(self.get_perception(state))
        currentPosition = self.getAvatarCoordinates(state)
        if not self.gotTheKey:
            self.keyPosition = self.getKeyPosition(state)
        self.exitPosition = self.getExitPosition(state)
        if self.lastState is not None:
            reward = self.getReward(self.lastState, currentPosition, state)
            self.replayMemory.pushExperience(Experience(
                self.lastState, self.lastActionIndex, reward, state))
            # Train
            loss = self.train()
            self.averageLoss += loss
            self.averageReward += reward

        index = self.getNextAction(state)
        # action = state.availableActions[index]
        self.lastState = state
        self.lastPosition = currentPosition
        if index is not None:
            self.lastActionIndex = index
        # print("Action and index: " + str(action) + " " + str(index))
        self.steps += 1
        return index

    def getElementCoordinates(self, state, element):
        result = None
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == element:
                    result = [i, j]
        return result

    def getAvatarCoordinates(self, state):
        return self.getElementCoordinates(state, 1.0)

    def getKeyPosition(self, state):
        return self.getElementCoordinates(state, 2.0)

    def getExitPosition(self, state):
        return self.getElementCoordinates(state, 3.0)

    def getDistanceToKey(self, state):
        return distance.cityblock(self.getAvatarCoordinates(state), self.keyPosition)

    def getDistanceToExit(self, state):
        return distance.cityblock(self.getAvatarCoordinates(state), self.exitPosition)

    def isCloserToKey(self, previousState, currentState):
        return self.getDistanceToKey(currentState) < self.getDistanceToKey(previousState)

    def isCloserToExit(self, previousState, currentState):
        return self.getDistanceToExit(currentState) < self.getDistanceToExit(previousState)

    def getReward(self, lastState, currentPosition, currentState):
        level = lastState
        col = int(currentPosition[0])  # col
        row = int(currentPosition[1])  # row
        reward = 0.0
        # if currentState.NPCPositionsNum < lastState.NPCPositionsNum:
        #     print('KILLED AN ENEMY')
        #     reward += 1.0
        if self.keyPosition is not None and self.isCloserToKey(lastState, currentState):
            reward += 3.0
        # if not self.isCloserToKey(lastState, currentState):
        #     reward += -1.0
        # if self.gotTheKey and self.isCloserToExit(lastState, currentState):
        #     print('Got the key and closer!')
        #     reward += 2.0
        if level[col][row] == 2.:
            # If we got the key
            print('GOT THE KEY')
            self.gotTheKey = True
            reward += 100.0
        elif level[col][row] == 3. and self.gotTheKey:
            # If we are at the exit
            print('WON')
            reward += 10000.0
        elif level[col][row] == 4.:
            # If we touched an enemy
            reward += -10.0
        elif level[col][row] == 0. or level[col][row] == 5.:
            # If we are in a safe spot or didn't move
            reward += -5.0
        return reward

        # Modify here to alter network inputs, be careful of dynamic arrays and to change network inputs
    def buildNetworkInput(self, state):
        perception = []
        perception = np.append(
            perception, np.ravel(state))
        # perception = np.append(perception, state.gameScore)
        # perception = np.append(perception, 0.0 if state.isGameOver else 1.0)
        # perception = np.append(perception, 0.0 if not self.gotTheKey else 1.0)
        # perception = np.append(perception, 0.0 if not self.closerToExit else 1.0)
        # perception = np.append(perception, 0.0 if not self.closerToKey else 1.0)
        # perception = np.append(perception, actionToFloat[state.avatarLastAction])
        # perception = np.append(perception, np.ravel(state.avatarOrientation))
        # perception = np.append(perception, len(state.NPCPositions)) # number of enemies
        # perception = np.append(perception, np.ravel([i.getPositionAsArray() for i in np.ravel(state.portalsPositions)]))
        # perception = np.append(perception, np.ravel(
        # [i.getPositionAsArray() for i in np.ravel(state.NPCPositions)]))
        # perception = np.append(perception, self.getDistanceToKey(state))
        # perception = np.append(perception, self.getDistanceToExit(state))
        # perception = np.append(perception, np.ravel(
        #     [i.getPositionAsArray() for i in np.ravel(state.resourcesPositions)]))
        return perception

    def getNextAction(self, state):
        # Do exploration or exploitation
        if self.movementStrategy.shouldExploit():
            #Exploitation
            # print('Exploitation')
            sd = tf.reshape(self.policyNetwork(tf.convert_to_tensor(
                [self.buildNetworkInput(state)], dtype=tf.float32)), (1, -1))
            return np.argmax(sd)
        else:
            #Exploration
            # print('Exploration')
            return random.randint(0, NUM_ACTIONS - 1)

    def train(self):
        if self.replayMemory.numSamples < batch_size * 3:
            return 0
        batch = self.replayMemory.sample(batch_size)
        # rawStates = [np.ravel(self.get_perception(val.state)) for val in batch]
        rawStates = [self.buildNetworkInput(val.state) for val in batch]
        states = tf.convert_to_tensor(rawStates, dtype=tf.float32)
        actions = np.array([val.actionIndex for val in batch])
        rewards = np.array([val.reward for val in batch])
        rawNextStates = [
            (np.zeros(state_size) if val.nextState is None else self.buildNetworkInput(val.nextState)) for val in batch]
        # rawNextStates = [(np.zeros(state_size) if val.nextState is None else val.nextState) for val in batch]
        # preTensorNextStates = [self.buildNetworkInput(val.nextState) for val in rawNextStates]
        nextStates = tf.convert_to_tensor(rawNextStates, dtype=tf.float32)
        # predict Q(s,a) given the batch of states
        prim_qt = self.policyNetwork(states)
        # predict Q(s',a') from the evaluation network
        prim_qtp1 = self.policyNetwork(nextStates)
        # copy the prim_qt into the target_q tensor - we then will update one index corresponding to the max action
        target_q = prim_qt.numpy()
        updates = rewards
        valid_idxs = np.array(nextStates).sum(axis=1) != 0
        batch_idxs = np.arange(batch_size)

        prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
        q_from_target = self.targetNetwork(nextStates)
        updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs],
                                                             prim_action_tp1[valid_idxs]]
        target_q[batch_idxs, actions] = updates
        loss = self.policyNetwork.train_on_batch(states, target_q)
        # update target network parameters slowly from policy network
        for t, e in zip(self.targetNetwork.trainable_variables, self.policyNetwork.trainable_variables):
            t.assign(t * (1 - TAU) + e * TAU)
        return loss

    def result(self, sso, gameWinner):
        self.gameOver = True
        reward = 0
        if self.lastActionIndex is not None:
            if gameWinner == 'PLAYER_LOSES':
                reward += -10.0
            elif gameWinner == 'PLAYER_WINS':
                reward += 10000.0
            self.replayMemory.pushExperience(Experience(
                self.lastState, self.lastActionIndex, reward, sso))
        self.episode += 1

        if self.gameOver:
            self.averageLoss /= self.steps
            print("Episode: {}, Reward: {}, avg loss: {}, eps: {}".format(
                self.episode, self.averageReward, self.averageLoss, self.movementStrategy.epsilon))
            print("Winner: {}".format(gameWinner))
            with train_writer.as_default():
                tf.summary.scalar(
                    'reward', self.averageReward, step=self.steps)
                tf.summary.scalar(
                    'avg loss', self.averageLoss, step=self.steps)
        if self.episode % 10 == 0:
            self.policyNetwork.save_weights("./network/zelda-ddqn.h5")
            print('Model saved!')
        return random.randint(0, 2)
