import numpy as np
import cv2
import mss
import time

###################################################################
###   Determing speed using 1-NN clustering
###################################################################

def load_digits():
  digits = [cv2.imread('digits/'+str(f)+'.png', 0) for f in range(10)]
  digits = np.array( digits )
  return digits

def get_speed(img, digits):
    # convert digits to grayscale
    #img1 = cv2.cvtColor(img[730:, 1265:1292], cv2.COLOR_BGR2GRAY) # get 1st digit of speed as black and white
    #img2 = cv2.cvtColor(img[730:, 1295:1322], cv2.COLOR_BGR2GRAY) # get 2nd digit of speed as black and white
    #img3 = cv2.cvtColor(img[730:, 1325:1352], cv2.COLOR_BGR2GRAY) # get 3rd digit of speed as black and white

    # negatticve indexing so i can alter input image size
    img1 = img[-38:, -101:-74] # get 1st digit of speed 
    img2 = img[-38:,  -71:-44] # get 2nd digit of speed 
    img3 = img[-38:,  -41:-14] # get 3rd digit of speed 

    # normalize to get better black and white
    img1[img1 > 250] = 255
    img1[img1 <= 250] = 0

    img2[img2 > 250] = 255
    img2[img2 <= 250] = 0

    img3[img3 > 250] = 255
    img3[img3 <= 250] = 0

    # norminaliz images
    num1, num2, num3 = 0,0,0
    best1, best2, best3 = 100000, 100000, 100000
    for idx, num in enumerate(digits):
        if np.sum(np.bitwise_xor(img1, num)) < best1:
            best1 = np.sum(np.bitwise_xor(img1, num))
            num1 = idx
        if np.sum(np.bitwise_xor(img2, num)) < best2:
            best2 = np.sum(np.bitwise_xor(img2, num))
            num2 = idx
        if np.sum(np.bitwise_xor(img3, num)) < best3:
            best3 = np.sum(np.bitwise_xor(img3, num))
            num3 = idx
        if np.max(img1) == 0:
            best1, num1 = 0, 0
        if np.max(img2) == 0:
            best2, num2 = 0, 0
        if np.max(img3) == 0:
            best3, num3 = 0, 0
    return float(100 * num1 + 10 * num2 + num3)

digits = load_digits()
def grab_img_and_speed(monitor, sct, resize_width, resize_height):
  img = np.array(sct.grab( monitor ), dtype=np.uint8) # convert to numpy array
  img = np.flip(img[:, :, :3], 2)     # convert to RGB numpy array
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to black and white

  speed = get_speed(img, digits)
  img = cv2.resize(img, (resize_width, resize_height))
  return img, speed

"""
Game controller adaptaed from
https://github.com/SamiKauhala/self-driving-car-in-tmnf/blob/master/keys.py
"""

import time
import pynput

def gamecontroller(action, controller=pynput.keyboard.Controller(), press_duration=0.006):
  selected_action =' '
  if action == 0: # forward
    ress_duration = 0.0075
    #controller.press(pynput.keyboard.Key.up)
    #controller.release(pynput.keyboard.Key.left)
    #controller.release(pynput.keyboard.Key.right)
    #controller.release(pynput.keyboard.Key.down)
    #time.sleep(press_duration)
    selected_action ='forward'

  elif action == 1: #left
    #controller.press(pynput.keyboard.Key.left)
    #controller.release(pynput.keyboard.Key.right)
    #controller.release(pynput.keyboard.Key.up)
    #controller.release(pynput.keyboard.Key.down)
    #time.sleep(press_duration)
    selected_action ='left'

  elif action == 2: #right
    #controller.press(pynput.keyboard.Key.right)
    #controller.release(pynput.keyboard.Key.left)
    #controller.release(pynput.keyboard.Key.up)
    #controller.release(pynput.keyboard.Key.down)
    #time.sleep(press_duration)
    selected_action ='right'

  elif action == 3: # reverse
    #controller.press(pynput.keyboard.Key.down)
    #controller.release(pynput.keyboard.Key.up)
    #controller.release(pynput.keyboard.Key.right)
    #controller.release(pynput.keyboard.Key.left)
    #time.sleep(press_duration)
    selected_action ='reverse'

  elif action == 4: # forward left
    #controller.press(pynput.keyboard.Key.up)
    #controller.press(pynput.keyboard.Key.left)
    #controller.release(pynput.keyboard.Key.right)
    #controller.release(pynput.keyboard.Key.down)
    #time.sleep(press_duration)
    #controller.release(pynput.keyboard.Key.left)
    selected_action ='forward left'

  elif action == 5: # forward right
    #controller.press(pynput.keyboard.Key.up)
    #controller.press(pynput.keyboard.Key.right)
    #controller.release(pynput.keyboard.Key.down)
    #controller.release(pynput.keyboard.Key.left)
    #time.sleep(press_duration)
    #controller.release(pynput.keyboard.Key.right)
    selected_action ='forward right'

  elif action == 6: # reverse left
    #controller.press(pynput.keyboard.Key.down)
    #controller.press(pynput.keyboard.Key.left)
    #controller.release(pynput.keyboard.Key.up)
    #controller.release(pynput.keyboard.Key.right)
    #time.sleep(press_duration)
    #controller.release(pynput.keyboard.Key.left)
    selected_action ='reverse left'

  elif action == 7: # reverse right
    #controller.press(pynput.keyboard.Key.down)
    #controller.press(pynput.keyboard.Key.right)
    #controller.release(pynput.keyboard.Key.left)
    #controller.release(pynput.keyboard.Key.up)
    #time.sleep(press_duration)
    #controller.release(pynput.keyboard.Key.right)
    selected_action = 'reverse right'

  elif action == 8: # do nothing
    #controller.release(pynput.keyboard.Key.down)
    #controller.release(pynput.keyboard.Key.up)
    #controller.release(pynput.keyboard.Key.left)
    #controller.release(pynput.keyboard.Key.right)
    #controller.release(pynput.keyboard.Key.up)
    selected_action = 'Do nothing'

  return selected_action


