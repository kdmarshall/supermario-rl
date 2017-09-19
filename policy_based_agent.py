import sys
import gym
import time
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

lr = 0.8
gamma = 0.95

env = gym.make('SuperMarioBros-1-1-v0')

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

class Agent(object):
    def __init__(self, lr, s_size, a_size, h_size):
        self.state_in = tf.placeholder(shape=[None, s_size],dtype=tf.int8)
        _input_node = tf.cast(self.state_in, dtype=tf.float32)
        _input_node_scaled = tf.div(_input_node, 255.)
        hidden = slim.fully_connected(_input_node_scaled, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
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

myAgent = Agent(lr=1e-2, s_size=172032, a_size=6, h_size=8) #Load the agent.

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

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
        s = env.reset() #57344
        s = np.reshape(s, (172032,))#np.reshape(s, (57344, 3))
        # s = s.T # (3, 57344)
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)
            action = one_hot(a)
            time.sleep(0.075)
            s1,r,d,_ = env.step(action) #Get our reward for taking an action given a bandit.
            s1 = np.reshape(s1, (172032,))#np.reshape(s1, (57344, 3)).T
            ep_history.append([s,action,r,s1])
            s = s1
            running_reward += r
            if d == True:
                print("Episode %s done."%i)
                #Update the network.
                ep_history = np.array(ep_history)
                print(ep_history.shape)
                break
                # ep_history[:,2] = discount_rewards(ep_history[:,2])
                # feed_dict = {myAgent.reward_holder:ep_history[:,2],
                #              myAgent.action_holder:ep_history[:,1],
                #              myAgent.state_in:np.vstack(ep_history[:,0])}
                # grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                # for idx,grad in enumerate(grads):
                #     gradBuffer[idx] += grad
        # if i % 100 == 0:
        #     print(np.mean(total_reward[-100:]))
        i += 1
