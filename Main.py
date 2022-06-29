import SettingData
import TrainModel
import CreateNoiseImages
import LoadModel

mainPath = 'C:/'
origPath = mainPath + 'BaseDeImagens/OriginalRuidos/'
gaussPath = mainPath + 'BaseDeImagens/Corrompidas/Gaussiano/'
saltNPepperPath = mainPath + 'BaseDeImagens/Corrompidas/SalEPimenta/'
artifactsPath = mainPath + 'BaseDeImagens/Corrompidas/Artefatos/'

inpFolderName = "Entrada/"
expFolderName = "Esperado/"

testPathGauss = mainPath + 'BaseDeImagens/TesteFinal/Gaussiano/'
testPathSP = mainPath + 'BaseDeImagens/TesteFinal/SP/'
testPathArt = mainPath + 'BaseDeImagens/TesteFinal/Artefatos/'

resultsPath = mainPath + 'BaseDeImagens/Resultados/'
resultsImgPath = mainPath + 'BaseDeImagens/Resultados/Imagens/'

predict_path = mainPath+ "BaseDeImagens/Resultados/Testes/G_SP/Teste 40k/"
predict_path_imgs = mainPath + "BaseDeImagens/Resultados/Testes/G_SP/Teste 40k/Imagens/"
#predict_path = resultsPath
#predict_path_imgs = resultsImgPath

FINAL_TEST_NOISY_COUNT = 90

SIZE = 256
PATCH_SIZE = 50
PATCH_STRIDE = 25

#35 (10170)
#25 (20700)
#20 (30060)
#17 (40860)
#15 (51030)
#13 (62640)
#12 (78300)
#11 (91710)
#10 (110430)



BATCH_SIZE = 32
trainEpochs = 150
augInc = 0

TEST_IMG_SIZE = 256
executionType = "P"
includeDataType = "G_SP"

MODEL_NAME = "model.h5"

camIterm = 18

def GetDataTrainTypes(includeDataType):
    
    dataPaths = []
    testDataPaths = []
    dataTypes = []
    if includeDataType == "G":
        dataPaths = [gaussPath]
        testDataPaths = [testPathGauss]
        dataTypes = [0]
    elif includeDataType == "SP":
        dataPaths = [saltNPepperPath]
        testDataPaths = [testPathSP]
        dataTypes = [1]
    elif includeDataType == "A":
        dataPaths = [artifactsPath]
        testDataPaths = [testPathArt]
        dataTypes = [2]
    elif includeDataType == "G_SP":
        dataPaths = [gaussPath,saltNPepperPath]
        testDataPaths = [testPathGauss,testPathSP]
        dataTypes = [0,1]
    elif includeDataType == "G_SP_A":
        dataPaths = [gaussPath,saltNPepperPath,artifactsPath]
        testDataPaths = [testPathGauss,testPathSP,testPathArt]
        dataTypes = [0,1,2]
    
    return dataPaths,testDataPaths,dataTypes


if executionType == "CN" or executionType == "CN_T":
    
    CreateNoiseImages.SaveNoisyImagesDataSet(
        origPath = origPath, 
        savePaths = [gaussPath,saltNPepperPath], 
        finalTestPaths = [testPathGauss,testPathSP], 
        tstImgsCount = FINAL_TEST_NOISY_COUNT,
        inpFolderName = inpFolderName, 
        expFolderName = expFolderName,
        width = SIZE,
        height = SIZE,
        widthFT = TEST_IMG_SIZE,
        heightFT = TEST_IMG_SIZE)

if executionType == "T" or executionType == "CN_T":
    
    dataPaths,testDataPaths,dataTypes = GetDataTrainTypes(includeDataType)
    
    SettingData.ReadImagesDataSet(paths = dataPaths,
                      inpFolderName = inpFolderName,
                      expFolderName = expFolderName, 
                      dataTypes = dataTypes, 
                      width = SIZE,
                      height = SIZE,
                      augInc = augInc,
                      patch_size = PATCH_SIZE,
                      patch_stride = PATCH_STRIDE)
    
    TrainModel.Train(
                     noisy = SettingData.noisyImages, 
                     original = SettingData.originalImages,
                     epochs = trainEpochs,
                     batchSize = BATCH_SIZE,
                     camIterm = camIterm, 
                     width = PATCH_SIZE, 
                     height = PATCH_SIZE,
                     saveModelPath = resultsPath
                     )

    LoadModel.FinalTestModel(modelName = MODEL_NAME, 
                             testPaths = testDataPaths, 
                             inpFolderName = inpFolderName, 
                             expFolderName = expFolderName, 
                             width = TEST_IMG_SIZE, 
                             height = TEST_IMG_SIZE,
                             resultsPath = resultsPath,
                             resultsImgPath = resultsImgPath)
    
elif executionType == "P":
    
    dataPaths,testDataPaths,dataTypes = GetDataTrainTypes(includeDataType)
    
    LoadModel.FinalTestModel(modelName = MODEL_NAME, 
                             testPaths = testDataPaths, 
                             inpFolderName = inpFolderName, 
                             expFolderName = expFolderName, 
                             width = TEST_IMG_SIZE, 
                             height = TEST_IMG_SIZE,
                             resultsPath = predict_path,
                             resultsImgPath = predict_path_imgs)
