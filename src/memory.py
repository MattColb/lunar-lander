import random
from collections import deque
class RecallMemory:
    def __init__(self, settings):
        self.settings = settings
        self.memory = deque(maxlen=settings.buffer_size)

    def __len__(self):
        return len(self.memory)
    
    def add(self, record):
        self.memory.append(record)

    def sample(self):
        sampled_items = random.sample(self.memory, self.settings.batch_size)
        return sampled_items