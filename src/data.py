import cv2
import numpy as np
import os
data = {
    "skulls": [[168, 16, 181, 33], [168, 56, 181, 73], [168, 96, 181, 113], [168, 136, 181, 153], [168, 176, 181, 193]],
    "signs": [[20, 1, 38, 15], [20, 41, 38, 55], [20, 81, 38, 95], [20, 121, 38, 135], [20, 161, 39, 175]],
    "roi": [0, 180, 190, 380]
}

def get_maped_boxes(factor, referencePoint):
    global data
    result_data = {
        "skulls": add_point_to_box_array(np.round(np.array(data["skulls"]) * factor), referencePoint).astype(int),
        "signs": add_point_to_box_array(np.round(np.array(data["signs"]) * factor), referencePoint).astype(int),
        "roi": add_point_to_box(np.round(np.array(data["roi"]) * factor), referencePoint).astype(int)
    }
    return result_data

def get_templates(texture_factor, width, height):
    result = {}
    if texture_factor > 1:
        result["signs"] = build_dict_in_folder("res/images/signs/high", 1)
    else:
        result["signs"] = build_dict_in_folder("res/images/signs/low", texture_factor)
    result["skull"] = get_factored_image("res/images/skull.png", texture_factor)
    return result

def build_dict_in_folder(folder, texture_factor):
    directory = os.fsencode(folder)
    result = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        img = cv2.imread(os.path.join(folder, filename))
        if texture_factor != 1:
            img = cv2.resize(img, np.round(np.array(img.shape[:2]) * texture_factor).astype(int)[::-1])
        result[filename] = img
    return result

def get_factored_image(image, factor):
    img = cv2.imread(image)
    if factor != 1:
        img = cv2.resize(img, np.round(np.array(img.shape[:2]) * factor).astype(int)[::-1])
    return img

def add_point_to_box_array(array, point):
    result = []
    for box in array:
        result.append(add_point_to_box(box, point))
    return np.array(result)

def add_point_to_box(box, point):
    return np.array([box[0] + point[0], box[1] + point[1], box[2] + point[0], box[3] + point[1]])