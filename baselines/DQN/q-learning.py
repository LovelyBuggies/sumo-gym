import random

class DefinedReplayBuffer(object):

    def __init__(self):
        self.memory = {
            (0, 0, 0, -100), (0, 1, 2, 100), (1, 0, 0, -100), (1, 0, 1, 50),
            (1, 1, 2, 20), (2, 0, 1/2, 50), (2, 1, 2, -20)
        }

    def sample(self, batch_size):
        dummy_memory = set([tuple(m) for m in self.memory[:-1]])
        return random.sample(dummy_memory, batch_size)


