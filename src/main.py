import sys
from PIL import ImageGrab
import cv2
import numpy as np
from rectangle import Rectangle
import time
import requests
import configparser
from data import get_maped_boxes, get_templates
import detect
import os

def find_rects(img, template, sens):

    w = template.shape[0]
    h = template.shape[1]

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= sens)
    rects = []
    for pt in zip(*loc[::-1]):
        rects.append(Rectangle(*pt, pt[0]+w, pt[1] + h))
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    return reduceRects(rects)

    

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


def calculate_aspect(width: int, height: int):
    def gcd(a, b):
        return a if b == 0 else gcd(b, a % b)

    r = gcd(width, height)
    x = int(width / r)
    y = int(height / r)

    return x , y
    

def loop(boxes, templates, debug):
    global previous_alive
    dead = run_detection(boxes, templates, debug)
    if(dead != None):
        alive = 5 - dead 
        if alive != previous_alive:
            print("Currently alive: " + str(alive), end="\r")
            debug.append("Currently alive: " + str(alive))
            requests.post("http://173.212.247.39:8000/report", json={"alive": alive, "group": group_num})
            previous_alive = alive

previous_alive = -1

def get_nw_resolution(config):
    if config['DEFAULT']['fullscreen'] == 'True':
        img = ImageGrab.grab()
        return img.width, img.height
    else:
        resoultion = config['WINDOWED']['nwResolution'].split("x")
        return int(resoultion[0]), int(resoultion[1])

def get_texture_factor(width, height):
    x, y = calculate_aspect(width, height)
    if x/y <= 16/9:
        return width / 1920 
    else:
        return height / 1080

def setup(config):
    referencePoint = [0, 0]
    width, height = get_nw_resolution(config)
    texture_factor = get_texture_factor(width, height)
    if config['DEFAULT']['fullscreen'] == 'False':
        referencePointStr = config['WINDOWED']['referencePoint'].split()
        referencePoint =  int(referencePointStr[0]), int(referencePointStr[1])  
    boxes = get_maped_boxes(texture_factor, referencePoint)
    templates = get_templates(texture_factor, width, height)
    return boxes, templates

def draw_box(img, box):
    cv2.rectangle(img, box[:2], box[2:4], (0,0,255))

def crop(img, box):
    return img[box[1]:box[3], box[0]:box[2]]

def debug_print(boxes):
    img = ImageGrab.grab().convert('RGB')
    cv_image = np.array(img)
    cv_image = cv_image[:, :, ::-1].copy()
    y_offset = boxes["roi"][1] 
    draw_box(cv_image, boxes["roi"])
    for box in boxes["skulls"]:
        box = [box[0], box[1] + y_offset, box[2], box[3] + y_offset]
        draw_box(cv_image, box)
    for box in boxes["signs"]:
        box = [box[0], box[1] + y_offset, box[2], box[3] + y_offset]
        draw_box(cv_image, box)
    cv2.imwrite("result.png", cv_image)

def get_cv2_screenshot(roi):
    image = ImageGrab.grab(bbox=list(roi)).convert('RGB')
    cv_image = np.array(image)
    return  cv_image[:, :, ::-1].copy() 

def read_cv2_image(path, roi):
    image = cv2.imread(path)
    return crop(image , roi)

def run_detection(boxes, templates, debug):
    img = get_cv2_screenshot(boxes["roi"])
    sign_confidence_values = np.array([])
    for box in boxes["signs"]:
        croped = crop(img, box)
        sign_confidence_values = np.append(sign_confidence_values, [np.fromiter(map(lambda template: detect.get_correlation(croped, templates["signs"][template]) , templates["signs"]), dtype=np.float64).max()])
    debug.append("Sign confidence values: \n" + np.array2string(sign_confidence_values))
    sign_confidence = np.average(sign_confidence_values)
    debug.append("average: " + str(sign_confidence))
    if sign_confidence < 0.6:
        return
    
    skull_confidence_values = np.fromiter(map(lambda box: detect.get_correlation(crop(img, box), templates["skull"]),boxes["skulls"]), dtype=np.float64)
    debug.append("Skull confidence values: \n" + np.array2string(skull_confidence_values))
    return len(np.where(skull_confidence_values > 0.6)[0])


def main():
    config = configparser.ConfigParser()
    config.read('res/config.ini')
    group_num = int(config['DEFAULT']['groupNum'])
    if not (group_num <= 10 and group_num >= 1):
        print("Please setup a valid group number in res/config.ini")
        time.sleep(5)
        sys.exit()
    
    boxes, templates = setup(config)
    print("Running...")
    while True:
        debug = []
        loop(boxes, templates, debug)
        if config['DEFAULT']['DEBUG'] == 'True':
            print("\n".join(debug))
        time.sleep(1)

if __name__ == '__main__':
    main()