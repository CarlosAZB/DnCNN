

from os import listdir
from os.path import splitext

import cv2 
import numpy as np
import DataAugmentation    

def AddSpeckleNoise(img, minProb = 0.15, maxProb = 0.35):
    
    row , col = img.shape
    
    pixelsPerc = np.random.uniform(minProb, maxProb)
    n_pixels = int(pixelsPerc * row * col)
    
    for i in range(n_pixels):
        y_coord=np.random.randint(0, row - 1)
        x_coord=np.random.randint(0, col - 1)
        
        if i < n_pixels/2:
            img[y_coord][x_coord] = 255
        else:
            img[y_coord][x_coord] = 0
    
    return img,pixelsPerc
    

def AddGaussianNoise(image, mean=0, minVar = 0.008, maxVar = 0.02):
    
    var = np.random.uniform(minVar, maxVar)
    
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    return out,var

def AddNoise(img, noiseType):
    if noiseType == 0:
        newImg,prob = AddGaussianNoise(img)
        return newImg,prob
    else:
        newImg,prob = AddSpeckleNoise(img)
        return newImg,prob
    
def ResizeImage(img,width,height):
    if len(img) != height or len(img[0]) != width:
        return cv2.resize(img,(width, height))
    
    return img
    

def AddNoiseAndSave(origPath,savePath,inpFolderName,expFolderName,noiseType,width,height, 
             imgsCount=0,isTest=False):
    
    originalImages = listdir(origPath)
    
    if imgsCount == 0:
        imgsCount = len(originalImages)
    
    for i in range(imgsCount):

        img = cv2.imread(origPath + originalImages[i],0)
        
        img = ResizeImage(img,width,height)
        filename, file_extension = splitext(originalImages[i])
        
        max_data_aug_inc = DataAugmentation.MAX_INCREMENT - 1
        var = np.random.randint(0, max_data_aug_inc)
        img = AddVariation(img,var)
        
        cv2.imwrite(savePath + expFolderName + str(i) + file_extension, img)
        
        noisyImg,prob = AddNoise(img, noiseType)
        
        imgName = str(i)
        if isTest:
            imgName += "-" + "{:.3f}".format(prob) 
        
        cv2.imwrite(savePath + inpFolderName + imgName + file_extension, noisyImg)

def AddVariation(img,var):
    
    return DataAugmentation.DataVariation(img, var)

def SaveNoisyImagesDataSet(origPath, savePaths, finalTestPaths, tstImgsCount,
                           inpFolderName, expFolderName,width,height,
                           widthFT,heightFT,noiseTypesCount=2):
    
    print("Creating data set...")
    
    for i in range(noiseTypesCount):
        AddNoiseAndSave(origPath,savePaths[i],inpFolderName,expFolderName,i,width,height)

    for i in range(noiseTypesCount):
        AddNoiseAndSave(origPath,finalTestPaths[i],inpFolderName,expFolderName,i,widthFT,heightFT,
                 tstImgsCount,True)
        
    

    
    
    
    

