
import numpy as np

from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Activation,Subtract
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,Callback
import time


def FormatImageGroup(imageGroup,width,height):
    return np.reshape(imageGroup, (len(imageGroup), height, width, 1)).astype('float32') / 255


class timecallback(Callback):
    def __init__(self,path):
        self.times = []
        self.timetaken = time.time()
        self.savePath = path
        self.fileName = "TrainTime.txt"
        self.epoch = 1
        
        f = open(path + self.fileName,'w')
        f.write('')
        f.close()
    def on_epoch_end(self,epoch,logs = {}):
        f = open(self.savePath + self.fileName,'a')
        f.write(str(self.epoch) + ";" + str(time.time() - self.timetaken) + '\n')
        f.close()
        self.epoch += 1


def CNNModel(camIterm):
    
    print("Creating model...")
    
    inpt = Input(shape=(None,None,1))

    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    
    for i in range(camIterm):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])
    model = Model(inputs=inpt, outputs=x)
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model
            

def Train(noisy, original, epochs, batchSize, camIterm,width,height,saveModelPath):
    
    noisy = FormatImageGroup(noisy,width,height)
    original = FormatImageGroup(original,width,height)
    
    model = CNNModel(camIterm)
    
    csv_logger = CSVLogger(saveModelPath + 'modelLog.txt')
    
    modelCheckpoint = ModelCheckpoint(
    filepath= saveModelPath + "/model.h5",
    save_weights_only=False,
    monitor='loss',
    verbose=0,
    save_best_only=True)
    
    timetaken = timecallback(saveModelPath)
    
    print("Training with " + str(len(noisy)) + " images...")
    model.fit(noisy, original, epochs = epochs, verbose = 1, batch_size = batchSize,
              callbacks=[modelCheckpoint, csv_logger,timetaken])
    
    return model


def TrainSKFold(noisy, original,filtersIterm, labels,k, epochs, batchSize, camIterm,width,height,saveModelPath):
    
    noisy = FormatImageGroup(noisy,width,height)
    original = FormatImageGroup(original,width,height)
    skf = StratifiedKFold(n_splits=k)
    
    kcount=1
    
    bestMetric = 0
    bestModel = None
    bestK = 0
    
    for train_index, test_index in skf.split(noisy, labels):
        
        noisy_train, noisy_test = noisy[train_index], noisy[test_index]
        expected_train, expected_test = original[train_index], original[test_index]
        model = CNNModel(camIterm,filtersIterm)
        
        csv_logger = CSVLogger(saveModelPath + 'modelLog.txt')
        
        modelCheckpoint = ModelCheckpoint(
        filepath= saveModelPath + "/Fold" + str(kcount) + "model_{epoch}.h5",
        save_weights_only=False,
        monitor='loss',
        verbose=0,
        save_best_only=False)
        
        print("Training with " + str(len(noisy_train)) + " images for " + str(kcount) + "-fold...\n\n")
        model.fit(noisy_train, expected_train, epochs = epochs, verbose = 1, batch_size = batchSize,
                  callbacks=[modelCheckpoint, csv_logger])
        
        modelEvaluate = model.evaluate(noisy_test, expected_test)

        if (modelEvaluate < bestMetric) or kcount == 1:
            bestMetric = modelEvaluate
            bestModel = model
            bestK = kcount
        
        endMsg = str(kcount) + "-Fold\n" + "MSE: " + str(modelEvaluate) + "\n\n"
        print(endMsg)
        
        f = open(saveModelPath + 'modelEvaluate.txt','a')
        f.write(endMsg)
        f.close()
        
        k+=1
    
    print("Choosin " + bestK + "-fold, with loss: " + bestMetric)
    return bestModel


        
        


