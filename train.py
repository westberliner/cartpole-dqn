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
env_name = 'CartPole-v1'

agent = Agent(model, memory, env_name)

for i in range(1000):
    print("GAME: {}".format(i))
    # play
    agent.play()

    # Test model every 100 games.
    if i > 0 and i%100 == 0:
        agent = Agent(model, memory, env_name)
        # agent.do_predicted_play()

    # learn
    agent.learn()

exit()
