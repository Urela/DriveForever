
from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT
import gym.spaces as spaces
import gym

from collections import deque
import numpy as np
import time
import cv2

from pynput.keyboard import Controller
import mss
import utils

# so frames are grayscale images ( 2D matrix ), easier to uses

class TMInterface(RealTimeGymInterface):
  def __init__(self, buffer_size=4, width=1366, height=768):
    self.mon= {"top": 0, "left": 0, "width": 1366, "height": 768 } # monitor 
    self.sct = mss.mss()

    self.last_time = time.time()
    self.digits = utils.load_digits()

    self.buffer_size = buffer_size
    self.buffer = deque(maxlen=buffer_size)
    self.keyboard = Controller()
    pass

  def send_control(self, control):
    if control is not None:
      utils.vertical(self.keyboard, control[0])   # acceleration   [-1, +1]
      utils.horizontal(self.keyboard, control[1]) # steering angle [-1, +1]
    pass

  def grab_img_and_speed(self):
    img = self.sct.grab( self.mon ) # get screen (as BGRA images)
    img = np.array(img, dtype=np.uint8) # convert to numpy array
    img = np.flip(img[:, :, :3], 2)     # convert to RGB numpy array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to black and white

    speed = utils.get_speed(img, self.digits)
    return img, speed

  def get_observation_space(self):
    speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
    frame = spaces.Box(low=0.0, high=255.0, shape=(self.buffer_size, self.mon["height"], self.mon["width"]))
    return spaces.Tuple((frame, speed))

  def get_action_space(self):
    return spaces.Box(low=-1.0, high=1.0, shape=(3, ))  # 1=f; -1=b; -1=l,+1=r

  def get_default_action(self):
    return np.array([0,0])    # [acceleration = 0, steering = 0]

  def get_obs_rew_done_info(self):
    img, speed = self.grab_img_and_speed()
    
    rew = speed
    self.buffer.append(img)
    imgs = np.array(list(self.buffer), dtype='float32')
    obs  = [imgs, speed]
    done = False  # TODO: True if race complete
    return obs, rew, done, {}
    #return [np.zeros((768,1366)),0], 0, False, {}

  def reset(self):
    self.send_control(self.get_default_action())
    # time.sleep(0.05)  # must be long enough for image to be refreshed
    img, speed = self.grab_img_and_speed()

    for _ in range(self.buffer_size):
        self.buffer.append(img)
    imgs = np.array(list(self.buffer), dtype='float32')
    obs  = [imgs, speed]
    return obs

  def wait(self):
    # perform default action when nothing is occuring, don't move
    self.send_control(self.get_default_action()) 
    pass

if __name__ == "__main__":
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
  #print(input_size)
  #print(output_size)

  # random test policy
  def model(obs): return np.random.uniform(-1, 1, (2, ))

  obs, rew, done = env.reset(), 0, False
  while not done:
      act = model(obs)
      obs, rew, done, info = env.step(act)
      #print(obs[1].shape)
      print(f"rew:{rew}")

