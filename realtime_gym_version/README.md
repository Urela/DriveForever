## DriveForever
This is gym environment for training reinforcemnet learning agents on TrackMania Nations Forever game. I built this as part of bigger project to build a self driving agent. I was inspired by Bluemax666 video - ( https://www.youtube.com/watch?v=yZFY5ZJtgyM )
## THIS HAS NOT BEEN TESTED, I NEED TO OPTIMIZE THIS,  ITS A MVP!!!!!

## Usage
The environment is continous and such environment can change without the agent interacting with the environment. To apply RL agents we need to force the environment is become a Markovian process. Or else the agent will start creating conspiracy theories. The environment is built on rtgym realtime gym, which * elastically constrain the times at which actions are sent and observations are retrieved, in a way that is transparent to the user* -https://github.com/yannbouteiller/rtgym

Import the envirnoment
```python
from TMenv import *
my_config = DEFAULT_CONFIG_DICT
my_config["interface"] = TMInterface
my_config["time_step_duration"] = 2      # TODO
my_config["start_obs_capture"] = 0.05                                                                        
my_config["time_step_timeout_factor"] = 1.0    
my_config["ep_max_length"] = 100                                                      
my_config["act_buf_len"] = 4                                                     
my_config["reset_act_buf"] = False                                                                                                    
                                                                                                                                                
env = gym.make("rtgym:real-time-gym-v0", config=my_config)

input_size  = env.observation_space[0].shape
output_size = env.action_space.shape
```
Run the envirnoment with your desired agent

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
     <img src="./Asserts/run1.png" />
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
