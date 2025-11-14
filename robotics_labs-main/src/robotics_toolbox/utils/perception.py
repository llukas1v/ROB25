#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  


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
    H = np.eye(3)

    # OK HW03: Detect circle in each image
    circle_centers = []                                     #array with circle center 
    for img in images:
        #basic mask for better circle recognition i use funcitno thet i found here https://stackoverflow.com/questions/10469235/opencv-apply-mask-to-a-color-image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        #mask for elimination od colors
        gray = cv2.medianBlur(gray, 9)                      #blure to reduce noice, paramer 9 I testing in test pach hw_data

        #circle detection, I aleso test parametrs on hw_data testbanch and especialy param1 was really reduce number of founded circles 
        circles = cv2.HoughCircles(                        
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=400,
            param2=30,
            minRadius=0,
            maxRadius=0
        )

        #detect circle center
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0, 0]                     # after discuss with my colleagues, we found that frist cicle in the array is most valid in our case. 
            circle_centers.append((x, y))               # take position of center

    # OK HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.
    if len(circle_centers) >= 4: # we have 8 Dof, so we need 8 or more referencis to compute homography
        cam_pos = np.array(circle_centers)
        
        all_real = []
        for pos in hoop_positions:
            act_real = pos['translation_vector'][:2] #we take, yust X,Y because Z we don't care. 
            all_real.append(act_real)
        
        
        if len(circle_centers) != len(all_real): #for case I  not recognize same number of circles as reals position form json 
            print("not enough circles recognize")
            return H
        real_pos = np.array(all_real)


        H,nothing  = cv2.findHomography(cam_pos, real_pos) #comutation of homography
    
    else: #if is not possible compute homography
        print("Not enough circle centers detected to compute homography.")

    return H
