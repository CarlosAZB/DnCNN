

from math import log10
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def RMSE(img1,img2):
    height = len(img1)
    width = len(img1[0])
    sum_ = 0
    
    for i in range(width):
        for j in range(height):
            sum_ += (img1[i][j] - img2[i][j])**2
            
    return (sum_ / (width*height))**.5


def PSNR(R,RMSE):
    return 20*log10(R/RMSE)



def ReadFolder(path,inputFolderName,expectedFolderName,height,width):
    
    inputImgs = os.listdir(path + inputFolderName)
    expectedImgs = os.listdir(path + expectedFolderName)
    
    input_data = []
    expected_data = []
    
    input_names = []
    
    for i in range(len(inputImgs)):
        imgNoisy = cv2.imread(path + inputFolderName + inputImgs[i],0)
        originalImg = cv2.imread(path + expectedFolderName + expectedImgs[i],0)
        
        input_names.append(inputImgs[i])
        
        
        if height == 0 or width == 0:
            height = len(imgNoisy)
            width = len(imgNoisy[0])
        elif len(imgNoisy) != height or len(imgNoisy[0]) != width:
            print("Resizing image...")
            imgNoisy = cv2.resize(imgNoisy,(width, height))
            originalImg = cv2.resize(originalImg,(width, height))
        
        input_data.append(imgNoisy)
        expected_data.append(originalImg)
        
    return input_data,expected_data,input_names
        

def FormatImageGroup(imageGroup,width,height):
    return np.reshape(imageGroup, (len(imageGroup), height, width, 1)).astype('float32') / 255

def LoadModel(resultsPath,modelName):
    print("Loagin model...")
    return load_model(resultsPath + modelName)

def PredictModel(model,input_data,expected_data,input_names,resultsPath,resultsImgPath,nameId):
    
    print("Predicting...")
    results = model.predict(input_data)
    
    f = open(resultsPath + str(nameId) + 'Valores.txt','w')
    
    psnrSum = 0
    rmseSum = 0
    resultsImgsPath = resultsImgPath + str(nameId) + "/"
    if not(os.path.exists(resultsImgsPath)):
        os.mkdir(resultsImgsPath)

    for i in range(len(results)):
        
        print("Processing result " + str(i+1))
        
        rmse = RMSE(results[i],expected_data[i])
        psnr = PSNR(1,rmse)
        
        psnrSum += psnr
        rmseSum += rmse
        
        savePath = resultsImgsPath + str(i) + 'Img/'
        
        if not(os.path.exists(savePath)):
            os.mkdir(savePath)
        
        cv2.imwrite(savePath + 'Resultado.jpeg', results[i] * 255)
        cv2.imwrite(savePath + 'Esperado.jpeg', expected_data[i] * 255)
        cv2.imwrite(savePath + 'Entrada' + input_names[i], input_data[i] * 255)
        f.write('Id: ' + str(i) + ' -- RMSE: ' + str(rmse) + ' -- PSNR: ' + str(psnr) + '\n')
    
    PSNRmean = psnrSum/len(results)
    RMSEmean = rmseSum/len(results)
    f.write('PSNR Mean: ' + str(PSNRmean) + '\n')
    f.write('RMSE Mean: ' + str(RMSEmean) + '\n')
    
    f.close()


def FinalTestModel(modelName,testPaths,inpFolderName,expFolderName,width,height,resultsPath,resultsImgPath):
        
    model = LoadModel(resultsPath,modelName)
        
    for i in range(len(testPaths)):
        testPath = testPaths[i]
        
        input_data,expected_data,input_names = ReadFolder(testPath,
                                                          inpFolderName,
                                                          expFolderName,
                                                          width,
                                                          height)
        
        input_data = FormatImageGroup(input_data,width,height)
        expected_data = FormatImageGroup(expected_data,width,height)
        
        PredictModel(model, input_data, expected_data,input_names,resultsPath,resultsImgPath,i)


    
    