import sys
import gym
import time
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

gamma = 0.95
epsilon = 0.35
env = gym.make('SuperMarioBros-1-1-v0')
observation_shape = list(env.observation_space.shape)

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def one_hot(index):
    encoding = [0]*6
    encoding[index] = 1
    return encoding

def get_random_action():
    choices = {
        'jump': [0, 0, 0, 0, 1, 0],
        'right_and_jump': [0, 0, 0, 1, 1, 0],
        'right': [0, 0, 0, 1, 0, 0]
    }
    choice = np.random.choice(['jump','right_and_jump','right'])
    return choices[choice]

class Agent(object):
    def __init__(self, lr, observ_shape, a_size):
        self.state_in = tf.placeholder(shape=[None]+observ_shape,dtype=tf.int8)
        _input_node = tf.cast(self.state_in, dtype=tf.float32)
        _input_node_scaled = tf.div(_input_node, 255.)
        h1 = tf.layers.conv2d(_input_node_scaled, 32, 8, 4, activation=tf.nn.relu)
        h2 = tf.layers.conv2d(h1, 64, 4, 2, activation=tf.nn.relu)
        h3 = tf.layers.conv2d(h2, 64, 3, 1, activation=tf.nn.relu)
        flat_h3 = tf.contrib.layers.flatten(h3)
        h4 = tf.layers.dense(flat_h3, 512, activation=tf.nn.relu)
        self.output = tf.layers.dense(h4, a_size, tf.nn.softmax)

        self.chosen_action = tf.argmax(self.output,1)
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

tf.reset_default_graph() #Clear the Tensorflow graph.

agent = Agent(3e-3, observation_shape, 6)
total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 600
update_frequency = 5
LVint=1

init = tf.global_variables_initializer()
# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []
        
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset() #(224, 256, 3) or 57344 flattened
        running_reward = 0
        ep_history = []
        print("Starting new episode.")
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(agent.output, feed_dict={agent.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)
            action = one_hot(a)
            time.sleep(0.075)
            # time.sleep(0.1)
            if np.random.choice([True, False], p=[epsilon, 1 - epsilon]):
                print("Random action")
                a = np.random.randint(6)
                action = one_hot(a)
                # action = get_random_action()
                # a = np.argmax(a)
            print(action)
            s1,r,d,_ = env.step(action) #Get our reward for taking an action
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r
            if j == (max_ep - 1):
                d = True
            if d == True:
                print("Episode %s done."%i)
                #Update the network.
                ep_history = np.array(ep_history)
                _ep_history = ep_history[:,2]
                ep_history[:,2] = discount_rewards(_ep_history)
                ep_arr = np.vstack(ep_history[:,0])
                ep_arr_reshaped = np.reshape(ep_arr, [j+1]+observation_shape)
                feed_dict = {
                            agent.reward_holder:ep_history[:,2],
                            agent.action_holder:ep_history[:,1],
                            agent.state_in:ep_arr_reshaped
                        }
                grads = sess.run(agent.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dict(zip(agent.gradient_holders, gradBuffer))
                    _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                
                total_reward.append(running_reward)
                total_length.append(j)
                env.change_level(new_level=LVint)
                break

        if i % 2 == 0:
            print("Mean total reward {}".format(np.mean(total_reward)))
        i += 1
