import numpy as np
from pprint import pprint

from model.dqn_model import DQNModel
from memory.deque_memory import DequeMemory
from agent.agent import Agent

# pprint(env.action_space)
# env.reset()
# action = env.action_space.sample()
# observation, reward, done, info = env.step(action)
# pprint(observation.shape)

IMAGE_SHAPE = (4,)
model = DQNModel(IMAGE_SHAPE)
memory = DequeMemory(1000)

agent = Agent(model, memory, 'CartPole-v0')

# play
agent.do_predicted_play(False)
# agent.do_random_play()

exit()
