import time
import cv2
import mss
import numpy as np
from pathlib import Path

def load_digits():
    #p = Path(os.path.dirname(os.path.realpath(__file__))) / 'digits'
    p = Path('digits')
    zero = cv2.imread(str(p / '0.png'), 0)
    One = cv2.imread(str(p / '1.png'), 0)
    Two = cv2.imread(str(p / '2.png'), 0)
    Three = cv2.imread(str(p / '3.png'), 0)
    four = cv2.imread(str(p / '4.png'), 0)
    five = cv2.imread(str(p / '5.png'), 0)
    six = cv2.imread(str(p / '6.png'), 0)
    seven = cv2.imread(str(p / '7.png'), 0)
    eight = cv2.imread(str(p / '8.png'), 0)
    nine = cv2.imread(str(p / '9.png'), 0)
    digits = np.array([zero, One, Two, Three, four, five, six, seven, eight, nine])
    return digits


def get_speed(img, digits):
    dig1 = np.array(img[730:, 1265:1292])  # get first digit of speed
    dig2 = np.array(img[730:, 1295:1322])  # get second digit of speed
    dig3 = np.array(img[730:, 1325:1352])  # get third digit of speed

    # convert digits to grayscale
    img1 = cv2.cvtColor(dig1, cv2.COLOR_BGR2GRAY)   
    img2 = cv2.cvtColor(dig2, cv2.COLOR_BGR2GRAY)   
    img3 = cv2.cvtColor(dig3, cv2.COLOR_BGR2GRAY)  

    img1[img1 > 250] = 255
    img1[img1 <= 250] = 0

    img2[img2 > 250] = 255
    img2[img2 <= 250] = 0

    img3[img3 > 250] = 255
    img3[img3 <= 250] = 0

    # norminaliz images
    num1=0
    num2=0
    num3=0

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

#print( get_speed(img, self.digits )) 
