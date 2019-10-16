import gym
import gc
import numpy as np
import random
import tensorflow as tf
from collections import deque
from pprint import pprint

class Agent:
    
    epsilon_min = 0.1
    epsilon_decay = 0.995
    epsilon = 1.0
    gamma = 0.95
    decay_state = 1.0
    batch_size = 32
    total_score_collection = []

    def __init__(self, model, memory, gym_type):
        self.model = model
        self.memory = memory
        self.gym_type = gym_type
        self.env = gym.make('CartPole-v0')

    def get_action(self, state, only_random=False):
        self.decay_state *= self.epsilon_decay
        explore_probability = max(self.decay_state, self.epsilon_min)

        if explore_probability > np.random.rand() or only_random:
            return self.env.action_space.sample(), "random"
        else:
            return self.model.predict(state), "predicted"

    def play(self):
        done = False
        episode_rewards = []
        state = self.env.reset()
        
        while done == False:
            #self.env.render()
            previous_state = state
            
            action, actionType = self.get_action(state)
            state, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)
            
            self.memory.append(previous_state, action, reward, state, done)

            if done:
                total_score = np.sum(episode_rewards)
                self.total_score_collection.append(total_score)
                print("Game finished with reward: {} results to an average of {}\n".format(total_score, np.mean(self.total_score_collection)))
                episode_rewards = []
                break

    def learn(self):
        length = 4
        targets = []
        batch = self.memory.get_batch(self.batch_size)
        previous_states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        current_states = np.array([each[3] for each in batch])
        game_finished = np.array([each[4] for each in batch])

        target_batch = []

        for i in range(0, len(batch)):
            done = game_finished[i]
            target = self.model.model.predict(np.array([previous_states[i]]))[0]

            if done:
                # "Punish" action for dqn if the game is over.
                target[actions[i]] = -rewards[i]
            else:
                # Reward future actions.
                predicted_next_qs = self.model.model.predict(np.array([current_states[i]]))[0]
                target[actions[i]] = rewards[i] + self.gamma * np.amax(predicted_next_qs)

            target_batch.append(target)

        targets = np.array([each for each in target_batch])
        
        self.model.fit(previous_states, targets)
        self.model.save()
        # Clear tensorflow session. Causes memory leak otherwise.
        # @see https://github.com/keras-team/keras/issues/13118
        tf.keras.backend.clear_session()

    def do_predicted_play(self, render=False, rounds=100):
        state = self.env.reset()
        done = False
        episode_rewards = []
        total_score_collection = []
        
        for i in range(rounds):
            print("Predicted Game Round: {}".format(i))
            while done == False:
                if render:
                    self.env.render()

                action = self.model.predict(state)
                state, reward, done, info = self.env.step(action)
                episode_rewards.append(reward)

                if done:
                    total_score = np.sum(episode_rewards)
                    total_score_collection.append(total_score)
                    print("Predicted Game finished with reward: {} results to an average of {}\n".format(total_score, np.mean(total_score_collection)))
                    done = False
                    self.env = gym.make('CartPole-v0')
                    state = self.env.reset()
                    # Clear tensorflow session. Causes memory leak otherwise.
                    # @see https://github.com/keras-team/keras/issues/13118
                    tf.keras.backend.clear_session()
                    episode_rewards = []
                    gc.collect()
                    break

    def do_random_play(self, render=False, rounds=100000):
        state = self.env.reset()
        done = False

        for i in range(rounds):
            while done == False:

                if render:
                    self.env.render()

                action, action_type = self.get_action(state)
                print("step {} of type {}\n".format(action, action_type))
                
                state, reward, done, info = self.env.step(action)

                if done:
                    done = False
                    state = self.env.reset()
                    gc.collect()


