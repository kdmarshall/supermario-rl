import sys
import gym
import random
import tensorflow as tf
import numpy as np

# num_actions = 6
lr = .8
gamma = .95
num_episodes = 2000

env = gym.make('SuperMarioBros-1-1-v0')
tf.reset_default_graph()
#Initialize table with all zeros
# Q = np.zeros([env.observation_space.n,env.action_space.n])
observation_shape = list(env.observation_space.shape)
num_actions = env.action_space.shape

def _qvalues(observ, reuse=False):
    """
        Takes in raw pixel values from
        the game and returns a dense representation
        with the shape == num_actions
    """
    with tf.variable_scope('qvalues', reuse=reuse):
        h1 = tf.layers.conv2d(observ, 32, 8, 4, activation=tf.nn.relu)
        h2 = tf.layers.conv2d(h1, 64, 4, 2, activation=tf.nn.relu)
        h3 = tf.layers.conv2d(h2, 64, 3, 1, activation=tf.nn.relu)
        flat_h3 = tf.contrib.layers.flatten(h3)
        h4 = tf.layers.dense(flat_h3, 512, activation=tf.nn.relu)
        return tf.layers.dense(h4, num_actions, None)

def Q(reward, next_observ):
    return reward + gamma * tf.reduce_max(_qvalues(next_observ), 1)

input_node = tf.placeholder(shape=[None]+observation_shape,dtype=tf.int8)
_input_node = tf.cast(input_node, dtype=tf.float32)
_input_node_scaled = tf.div(_input_node, 255.)
print(_input_node_scaled.get_shape())
actions = _qvalues(_input_node_scaled)
print(actions.get_shape())

