import random

from pprint import pprint

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def pushExperience(self, experience):
        if len(self.memory) < self.capacity:
            print(experience)
            self.memory.append(experience)

    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)
