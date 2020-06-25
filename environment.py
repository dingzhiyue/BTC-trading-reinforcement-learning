import gym

env = gym.make('Taxi-v2')
env.reset()

print(env.action_space.n)
#for i in range(6):
 #   act = env.action_space.sample()
  #  obs, rew, done, info = env.step(act)
   # env.render()