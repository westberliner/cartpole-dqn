# OpenAI Gym cartpole solved by a Neural Network (DQN) in Tensorflow 2

CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
Run `python play.py` to check. 

## Requirements

* Python 3.6
* Pipenv

## Installation

```
$ pipenv install
$ pipenv shell
```

## Create/Train/Play with Model

```
$ pipenv run python train.py
$ pipenv run python play.py
```

## Show log

```
$ tensorboard --logdir=logs
```
Currently I get an error: https://github.com/tensorflow/tensorflow/issues/32384

## Resources ##

* https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4
* https://fakabbir.github.io/reinforcement-learning/docs/doom/
