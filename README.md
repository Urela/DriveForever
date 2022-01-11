## DriveForever
This is a OpenAI gym interface for the TrackMania Nations Forever game. 

## Usage

```python
  import gym
  from TrackManiaEnv import TrackManiaEnv  
  
  env = TrackManiaEnv()
  obs = env.reset()
  for x in range(200):
    action = env.action_space.sample()    # Some random policy
    obs, reward, done, info = env.step(action)
    print(reward)
```
The environement can be vectorized, allowing for multiple instances of the game to be recorded if you can get multiple instances of the game running concurrently.
```python
  def make_env():
    def thunk():
      env = TrackManiaEnv()
      return env
    return thunk

  num_envs = 1
  env = gym.vector.SyncVectorEnv([make_env() for i in range(num_envs)])
```

### Envirnoment Config 
- Reward: Current implention has the reward function as the speed of the car. The faster laps require less crashes
- States: Four stacked frames sized 84x84 (Black and white images)
- Done: is always False ( we are treating this an non-episodeic environment)

<p align="center">
     <img src="./Asserts/run1.png" />
</p>

### Note
- Note this is a simple program to interact with TrackMania Nations Forever game to build a self driving car with ease. The game is free to download. Download the game and use the code to interact with the game
- This only works with linux as we using pynupt to process keypresses. Watch Sentdex GTA 5 tutorial (https://github.com/Sentdex/pygta5)  to learn how to use windows specific driver (should take 1 hour to understand)
- No offical API for this version of the game. So we use computer vision tricks to determine data. If you don't have specific reason, download Track Mania Stadium. Also free but the game has API for data. 
## Dependencies
- pynput
- mss
- gym
- opencv

## Note
- This is work is based on the tmrl https://pypi.org/project/tmrl/ project.
- Ported for linux only

## Reference 
- [Using Python programming to Play Grand Theft Auto 5](https://github.com/Sentdex/pygta5), Great starting point
- [TrackMania through Reinforcement Learning (tmrl)] (https://pypi.org/project/tmrl/)
