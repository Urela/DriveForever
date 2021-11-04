# adapted from tmrl 
# https://github.com/trackmania-rl/tmrl/blob/master/tmrl/custom/custom_gym_interfaces.py
from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT
from collections import deque
import gym.spaces as spaces
import gym
import numpy as np
import keys
import speed as velocity
import mss
import time
from pynput.keyboard import Controller
import cv2

class TrackManiaInterface(RealTimeGymInterface):
  def __init__(self, buffer_size=4):
    self.monitor = {"top": 0, "left": 0, "width": 1366, "height": 768 }
    self.sct = mss.mss()

    self.last_time = time.time()
    self.digits = velocity.load_digits()

    self.buffer_size = buffer_size
    self.buffer = deque(maxlen=buffer_size)
    self.keyboard = Controller()
    pass

  def send_control(self, control):
    if control is not None:
      keys.vertical(self.keyboard, control[0])   # acceleration   [-1, +1]
      keys.horizontal(self.keyboard, control[1]) # steering angle [-1, +1]
    pass

  def grab_img_and_speed(self):

    img = self.sct.grab( self.monitor ) # get screen (as BGRA images)
    img = np.array(img, dtype=np.uint8) # convert to numpy array
    img = np.flip(img[:, :, :3], 2)     # convert to RGB numpy array

    speed = velocity.get_speed(img, self.digits )
    return img, speed

  def reset(self):
    self.send_control(self.get_default_action())
    # time.sleep(0.05)  # must be long enough for image to be refreshed
    img, speed = self.grab_img_and_speed()

    for _ in range(self.buffer_size):
        self.buffer.append(img)
    imgs = np.array(list(self.buffer), dtype='float32')
    obs = [speed, imgs]
    return obs

  def get_observation_space(self):
    speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
    frame = spaces.Box(low=0.0, high=255.0, shape=(self.buffer_size, 50, 190, 3))
    return spaces.Tuple((speed, frame))

  def get_action_space(self):
    return spaces.Box(low=-1.0, high=1.0, shape=(3, ))  # 1=f; -1=b; -1=l,+1=r

  def get_default_action(self):
    return np.array([0,0])   # [acceleration = 0, steering = 0]

  def get_obs_rew_done_info(self):
    img, speed = self.grab_img_and_speed()
    
    rew = speed
    self.buffer.append(img)
    imgs = np.array(list(self.buffer), dtype='float32')
    obs = [speed, imgs]
    done = False  # TODO: True if race complete
    return obs, rew, done, {}


  def wait(self):
    self.send_control(self.get_default_action()) # perform default action when nothing is occuring, don't move
    pass

my_config = DEFAULT_CONFIG_DICT
my_config["interface"] = TrackManiaInterface
#my_config["time_step_duration"] = 0.2   # 
my_config["time_step_duration"] = 2
my_config["start_obs_capture"] = 0.05    # It defines the time at which an observation starts being retrieved
my_config["time_step_timeout_factor"] = 1.0 # maximum elasticity of the framework before a time-step times-out
my_config["ep_max_length"] = 100         # entry is the maximum length of an episode, 
                                         # When this number of time-steps have been performed since the last reset(). done = true
my_config["act_buf_len"] = 4             # entry is the size of the action buffer
my_config["reset_act_buf"] = False       # tells whether the action buffer should be reset with default actions when reset() is called
                                         # because calls to reset() only change the position of the target, and not the dynamics of the drone. 

env = gym.make("rtgym:real-time-gym-v0", config=my_config)
# random test policy
def model(obs):
  return np.random.uniform(-1, 1, (2, ))


done = False
obs = env.reset()
rew = 0
while not done:
    act = model(obs)
    obs, rew, done, info = env.step(act)
    print(f"rew:{rew}")
#
#print("Environment benchmarks:")
#pprint.pprint(env.benchmarks())

# TODO
# fix the time error thing  
#  - it probably means that your inference time is too long for the time-step you are trying to use.
#  - Done = True flag for winning race
