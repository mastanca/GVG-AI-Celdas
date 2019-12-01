import random

class EpsilonStrategy():
    def __init__(self):
        self.epsilon = 1
        self.epsilonDecreaseRate = 0.01

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def shouldExploit(self):
        if self.epsilon > 0:
            self.epsilon -= self.epsilonDecreaseRate
        return random.uniform(0, 1) > self.epsilon
