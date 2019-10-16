from collections import deque
import random

class DequeMemory:
    
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def append(self, old_observation, action, reward, observation, done):
        self.memory.append((old_observation, action, reward, observation, done))

    def get_batch(self, batch_size):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        # print('sample', batch)
        random.shuffle(batch)
        # print('shuffle', batch)
        return batch
