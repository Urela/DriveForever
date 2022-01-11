import gym
import logging
import numpy as np
from gym import spaces

from pynput.keyboard import Controller
import mss
import cv2
import utils

logger = logging.getLogger(__name__)

class TrackManiaEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, **kwargs):
    self.monitor = {"top": 0, "left": 0, "width": 1366, "height": 768 } # monitor
    self.sct = mss.mss()

    self.keyboard = Controller()
    self.digits = np.array( [cv2.imread('digits/'+str(f)+'.png', 0) for f in range(10)])

    # observation image size
    self.img_width = 84
    self.img_height = 84

    # Actions 1D vector
    self.action_space = spaces.Box( low=0, high=1, shape=(9,1), dtype=np.float16)

    # four frames stacked
    self.observation_space = spaces.Box(low=0, high=1,
        shape=(4,self.img_width, self.img_height), dtype=np.float16)
    pass

  def grab_img_and_speed(self):
    img = np.array(self.sct.grab( self.monitor ), dtype=np.uint8) # convert to numpy array
    img = np.flip(img[:, :, :3], 2)     # convert to RGB numpy array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to black and white

    speed = utils.get_speed(img, self.digits)
    img = cv2.resize(img, (self.img_width, self.img_height))
    return img, speed
	
  def step(self, action):
    img0, speed0 = self.grab_img_and_speed()
    img1, speed1 = self.grab_img_and_speed()
    img2, speed2 = self.grab_img_and_speed()
    img3, speed3 = self.grab_img_and_speed()

    obs = np.array([img0,img1,img2,img3], dtype='float32')
    reward = (speed0 + speed1 + speed2 + speed3)/4
    done = False  # contionus space , done is always false
    info = {}
    return obs, reward, done, info

  def reset(self):
    img, _ = self.grab_img_and_speed()
    return np.array([img,img,img,img], dtype='float32')

  def close(self):
    pass

  def render(self, mode='human', close=False):
    pass

if __name__ == "__main__":
  import time
  import gym
  from TrackManiaEnv import TrackManiaEnv  

  def make_env(idx, seed, record=False, run_name=''):
    def thunk():
      env = TrackManiaEnv()
      return env
    return thunk

  num_envs = 1
  seed = 1
  envs = gym.vector.SyncVectorEnv([
        make_env(i, seed+i, record=False, run_name=f"{int(time.time())}" ) 
            for i in range(num_envs)
        ])
  obs = envs.reset()
  for x in range(200):
    action = envs.action_space.sample()
    obs, reward, done, info = envs.step(action)
    #print(reward.shape)
    print(reward)
    #for item in info:
    #  if "episode" in item.keys():
    #    print(f"Episodic return: {item['episode']['r']}")
    #    # no need to obs = env.rest as gym vector does it automatically

