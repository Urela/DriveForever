import cv2
import time
import random
import numpy as np
from pynput.keyboard import Key, Controller

###################################################################
###   Control mechanism
###################################################################
def horizontal(controller, press_duration=0):
  if press_duration == 0:
    controller.release(Key.right)
    controller.release(Key.left)
  elif press_duration < 0:       # turn left if negative
    controller.press(Key.left)
    controller.release(Key.right)
    time.sleep(abs(press_duration))
    controller.release(Key.left)
  else:
    controller.press(Key.right)
    controller.release(Key.left)
    time.sleep(abs(press_duration))
    controller.release(Key.right)

def vertical(controller, press_duration=0):
  if press_duration ==0:
    controller.release(Key.down)
    controller.release(Key.up)
  elif press_duration < 0:       # reverse if negative
    controller.press(Key.down)
    controller.release(Key.up)
    time.sleep(abs(press_duration))
    controller.release(Key.down)
  else:
    controller.press(Key.up)
    controller.release(Key.down)
    time.sleep(abs(press_duration))
    controller.release(Key.up)

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


if __name__ == "__main__":
  print("Debugging ")

  ## test control mechanism 
  #keyboard = Controller()
  #for x in range(100):
  #  signal_x = random.uniform(-1, 1)
  #  signal_y = random.uniform(-1, 1)
  #  vertical(keyboard, signal_x)
  #  horizontal(keyboard, signal_y)

  #################################################

  import mss
  monitor = {"top": 0, "left": 0, "width": 1366, "height": 768 }
  digits = load_digits()
  sct = mss.mss()

  while True:
    img = np.asarray(sct.grab( monitor ))

    img = sct.grab( monitor ) # get screen (as BGRA images)
    img = np.array(img, dtype=np.uint8) # convert to numpy array
    img = np.flip(img[:, :, :3], 2)     # convert to RGB numpy array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to black and white

    #print( img.shape)
    print( get_speed(img, digits ))     #

