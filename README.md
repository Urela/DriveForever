## DriveForever
This is gym environment for training reinforcemnet learning agents on TrackMania Nations Forever game. I built this as part of bigger project to build a self driving agent. I was inspired by Bluemax666 video - ( https://www.youtube.com/watch?v=yZFY5ZJtgyM )

## Usage
The environment is built on rtgym realtime gym, which allows us to sync realtime action with frames (states) and their rewards. *is to elastically constrain the times at which actions are sent and observations are retrieved, in a way that is transparent to the user* -https://github.com/yannbouteiller/rtgym

```python
  # random test policy
  def model(obs): return np.random.uniform(-1, 1, (2, ))

  obs, rew, done = env.reset(), 0, False
  while not done:
      act = model(obs)
      obs, rew, done, info = env.step(act)
      #print(obs[1].shape)
      print(f"rew:{rew}")
```
### Envirnoment Config 
- Reward: Currently we are following the tmrl project, our reward is the speed of the vehicle, the faster you go the quicker your time will be
- States: So far states are just raw frames.
- Done: is always False ( we are treating this an non-episodeic environment)

<p align="center">
     <img src="./Asserts/run1.png">
</p>
### TODO
- Add a learning agent
- make program faster, right now each timestep takes 1 second.
- change reward function ( We are think game time, negative reward for time)

## Dependencies
- pynput
- rtgym
- gym
- opencv

## Note
- This is work is based on the tmrl https://pypi.org/project/tmrl/ project.
- Ported for linux only

## Reference 
[TrackMania through Reinforcement Learning (tmrl)](https://pypi.org/project/tmrl/), Great starting point
