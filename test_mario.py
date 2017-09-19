import gym
import multiprocessing
# multiprocessing_lock = multiprocessing.Lock()
# gym.error.Error: Env.configure has been removed in gym v0.8.0, released on 2017/03/05. If you need Env.configure, please use gym version 0.7.x from pip, or checkout the `gym:v0.7.4` tag from git
env = gym.make('SuperMarioBros-1-1-v0')
# env.configure(lock=multiprocessing_lock)
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    # action is a len 6 list of 0 or 1
    # action = [0, 0, 0, 1, 1, 0]    # [up, left, down, right, A, B]
    #  - An action of '1' represents a key down, and '0' a key up.
    # - To toggle the button, you must issue a key up, then a key down.
    print action
    observation, reward, done, info = env.step(action)
    # reward is a pos and neg float value
    print reward
