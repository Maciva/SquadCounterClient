import cv2 as cv

def get_correlation(img, template):
    res = cv.matchTemplate(img,template, cv.TM_CCOEFF_NORMED )
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return max_val
