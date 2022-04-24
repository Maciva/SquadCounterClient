import sys
from PIL import ImageGrab
import cv2
import numpy as np
from rectangle import Rectangle
import time
import requests
import configparser


def count_occurrences(img, template, sens):

    w = template.shape[0]
    h = template.shape[1]

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= sens)
    rects = []
    for pt in zip(*loc[::-1]):
        rects.append(Rectangle(*pt, pt[0]+w, pt[1] + h))
    return len(reduceRects(rects))

    

def reduceRects(rects):
    found = True
    while found:
        tmp = find_intersecting_rect(rects)
        if tmp:
            first = rects[tmp[0]]
            scnd = rects[tmp[1]]
            new = first & scnd
            rects.remove(first)
            rects.remove(scnd)
            rects.append(new)
        else:
            found = False
    return rects
                


def find_intersecting_rect(rects):
    for i in range(len(rects)):
            for j in range(len(rects))[i+1:]:
                tmp = rects[i] & rects[j]
                if tmp:
                    return (i,j)
    return None


def loop():
    global previous_alive
    image = ImageGrab.grab(bbox=(x1,y1,x2,y2)).convert('RGB')
    cv_image = np.array(image)
    cv_image = cv_image[:, :, ::-1].copy() 
    sign_count = count_occurrences(cv_image ,template_sign, 0.6)
    
    
    if sign_count:
        alive = 5 - count_occurrences(cv_image ,template, 0.8)
        if alive != previous_alive:
            print("Currently alive: " + str(alive), end="\r")
            requests.post("http://173.212.247.39:8000/report", json={"alive": alive, "group": group_num})
            previous_alive = alive

previous_alive = -1
template_sign = cv2.imread('res/sign.png')
template = cv2.imread('res/skull.png')
config = configparser.ConfigParser()
config.read('res/config.ini')
group_num = int(config['DEFAULT']['groupNum'])
if not (group_num <= 10 and group_num >= 1):
    print("Please setup a valid group number in res/config.ini")
    time.sleep(5)
    sys.exit()
    
x1 = int(config['DEFAULT']['x1'])
y1 = int(config['DEFAULT']['y1'])
x2 = int(config['DEFAULT']['x2'])
y2 = int(config['DEFAULT']['y2'])
image = ImageGrab.grab(bbox=(x1,y1,x2,y2)).convert('RGB')
image.save("../result.png")
print("Running...")
while True:
    loop()
    time.sleep(1)