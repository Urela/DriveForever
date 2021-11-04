import time
import random
from pynput.keyboard import Key, Controller

def horizontal(controller, press_duration=0):
  if press_duration < 0:              # go left if negative
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
  if press_duration < 0:              # reverse if negitve
    controller.press(Key.down)
    controller.release(Key.up)
    time.sleep(abs(press_duration))
    controller.release(Key.down)
  else:
    controller.press(Key.up)
    controller.release(Key.down)
    time.sleep(abs(press_duration))
    controller.release(Key.up)


# test keys 
if __name__ == "__main__":
  keyboard = Controller()
  for x in range(100):
    signal_x = random.uniform(-1, 1)
    signal_y = random.uniform(-1, 1)
    vertical(keyboard, signal_x)
    horizontal(keyboard, signal_y)
