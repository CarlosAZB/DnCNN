
import cv2
import numpy as np

MAX_INCREMENT = 6

def DataAug(img, augInc):
    imgs = []
    
    if augInc > MAX_INCREMENT:
        augInc = MAX_INCREMENT
    
    for i in range(augInc):
        imgs.append(DataVariation(img, i))

    return imgs

def DataVariation(img,variation):
    if variation == 0:
        return img
    elif variation == 1:
        return Rotate(img,90)
    elif variation == 2:
        return FlipImg(1, img)
    elif variation == 3:
        return Rotate(FlipImg(1, img),90)
    elif variation == 4:
        return Rotate(img,-90)
    elif variation == 5:
        return Rotate(FlipImg(1, img),-90)



def Rotate(image, angle, scale = 1.0):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def FlipImg(direction,img):
    if direction == 0:
        return cv2.flip(img, 0) #flip verticaly
    elif direction == 1:
        return cv2.flip(img, 1) #flip horizontal
    else:
        return img
    
def TranslateImg(img, dir_type, intensity):
    
    h, w = img.shape[:2]
    mov_height, mov_width = 0, 0
    
    if dir_type == 0:
        mov_width = w * intensity 
    if dir_type == 1:
        mov_height = h * intensity 
    elif dir_type == 2:
        mov_height, mov_width = w * intensity, h * intensity
    
    T = np.float32([[1, 0, mov_width], [0, 1, mov_height]])
    return cv2.warpAffine(img, T, (w, h))