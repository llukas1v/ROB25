#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa
import matplotlib.pyplot as plt


def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    """
    
    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)
    translations=[]
    s=[]
    for i,img_bgr in enumerate(images):
        translations.append(hoop_positions[i]["translation_vector"][:2])
        img = cv2.medianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),7)
        threshold=60
        color=[50, 95, 130]
        # Compute absolute difference from target color
        diff = np.abs(img - color)

        # Compute distance in RGB space
        dist = np.linalg.norm(diff, axis=2)

        # Threshold to create a binary mask
        mask = np.uint8((dist < threshold) * 255)
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1.2,              # Inverse ratio of accumulator resolution
            minDist=50,          # Minimum distance between circle centers
            param1=100,          # Higher threshold for Canny edge detector
            param2=20,           # Accumulator threshold for circle detection
            minRadius=100,        # Minimum circle radius
            maxRadius=500        # Maximum circle radius
        )
        x, y, r = circles[0, 0]
        s.append(np.array([x,y]))
    s = np.array(s)
    translations=np.array(translations)
    H , m = cv2.findHomography(s,translations,cv2.RANSAC)
    return H

"""
def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    translations = np.array([item['translation_vector'][:2] for item in hoop_positions])
    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)
    # todo HW03: Detect circle in each image
    s = []

    for imge in images:
        img = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB) 
        target_color = np.array([47, 90, 123])  # RGB 
        threshold = 50  # maximální vzdálenost (0–441)
        img = cv2.medianBlur(img,5)

        dist = np.linalg.norm(img - target_color, axis=2)
        mask = (dist < threshold).astype(np.uint8) * 255

        circles = cv2.HoughCircles(
            mask,                      # vstupní obraz (grayscale)
            cv2.HOUGH_GRADIENT,        # metoda
            dp=1.2,                    # inverzní poměr rozlišení akumulátoru
            minDist=50,                # minimální vzdálenost mezi kružnicemi
            param1=100,                # horní práh pro Cannyho detektor hran
            param2=30,                 # práh pro detekci kružnic (nižší = více, i falešných)
            minRadius=100,             # minimální poloměr            
            maxRadius= 300             # maximální poloměr
            )
        #print(circles)
        
        x, y, r = circles[0, 0]
        s.append(np.array([x,y]))

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)    # kružnice
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3)    # střed
            
            
            

        #cv2.imshow("Mask", img)
        #cv2.waitKey(0)
        
    #cv2.destroyAllWindows()
    
    
    # todo HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.
    
    s = np.array(s)
    H , m = cv2.findHomography(s,translations,cv2.RANSAC)


    return  H
    """