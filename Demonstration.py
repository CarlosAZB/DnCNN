import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

mainPath = "C:/TestImgs/"
noisyFolder = "noisy/"
modelName = "model.h5"
SIZE = 256


def ReadFolder(path,inputFolder,height,width):
    
    inputImgs = os.listdir(path + inputFolder)
    input_data = []
    input_names = []
    
    for i in range(len(inputImgs)):
        imgNoisy = cv2.imread(path + inputFolder + inputImgs[i],0)
        
        input_names.append(inputImgs[i])
        
        if height == 0 or width == 0:
            height = len(imgNoisy)
            width = len(imgNoisy[0])
        elif len(imgNoisy) != height or len(imgNoisy[0]) != width:
            print("Resizing image...")
            imgNoisy = cv2.resize(imgNoisy,(width, height))
        
        input_data.append(imgNoisy)
        
    return input_data,input_names
        

def FormatImageGroup(imageGroup,width,height):
    return np.reshape(imageGroup, (len(imageGroup), height, width, 1)).astype('float32') / 255

def LoadModel(resultsPath,modelName):
    print("Loagin model...")
    return load_model(resultsPath + modelName)

def PredictModel(model,main_path,input_data,input_names):
    
    print("Predicting...")
    results = model.predict(input_data)
    
    savePath = main_path + "/Results/"
    if not(os.path.exists(savePath)):
        os.mkdir(savePath)

    for i in range(len(results)):
        
        print("Processing result " + str(i+1))
        
        savePath_ = savePath + str(i) + 'Img/'
        
        if not(os.path.exists(savePath_)):
            os.mkdir(savePath_)
        
        cv2.imwrite(savePath_ + 'Resultado.jpeg', results[i] * 255)
        cv2.imwrite(savePath_ + 'Entrada' + input_names[i], input_data[i] * 255)



def ProcessImgs(mainPath,modelName,inpFolder,width,height):
    model = LoadModel(mainPath,modelName)
    input_data,input_names = ReadFolder(mainPath,inpFolder,width,height)
    input_data = FormatImageGroup(input_data,width,height)
    PredictModel(model,mainPath, input_data, input_names)



ProcessImgs(mainPath,modelName,noisyFolder,SIZE,SIZE)
    
    
