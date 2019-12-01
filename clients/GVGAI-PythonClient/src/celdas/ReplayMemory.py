import random

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def pushExperience(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)

    def sample(self):
        return random.sample(self.memory, 1)