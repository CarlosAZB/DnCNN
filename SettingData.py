
import os
import cv2
import numpy as np

import DataAugmentation as DataAugm

imgsType = []
noisyImages = []
originalImages = []

def AddImages(noisy,expected,label):
    global noisyImages,originalImages
    noisyImages += noisy
    originalImages += expected
    for i in range(len(noisy)):
        imgsType.append(label)
        

def ReadImagesDataSet(paths,inpFolderName,expFolderName, dataTypes, width,height,
                      augInc,patch_size,patch_stride):
    
    print('Setting data...')
    
    for i in range(len(paths)):
        inputImgs = os.listdir(paths[i] + inpFolderName)
        expectedImgs = os.listdir(paths[i] + expFolderName)
        
        for j in range(len(inputImgs)):
            imgNoisy = cv2.imread(paths[i] + inpFolderName + inputImgs[j],0)
            origImg = cv2.imread(paths[i] + expFolderName + expectedImgs[j],0)
    
            if len(imgNoisy) != height or len(imgNoisy[0]) != width:
                imgNoisy = cv2.resize(imgNoisy,(width, height))
                origImg = cv2.resize(origImg,(width, height))
            
            orig_patches,noisy_patches = GetPatches(imgNoisy,origImg,augInc,patch_size,patch_stride)
            AddImages(noisy_patches, orig_patches, dataTypes[i])


def GetPatches(imgNoisy,imgOrig,augInc,patch_size,stride):

    h, w = imgNoisy.shape
    scales = [1.0, 0.9, 0.8, 0.7]
    noisy_patches = []
    orig_patches = []

    max_data_aug_inc = DataAugm.MAX_INCREMENT - 1

    for s in scales:
        new_h, new_w = int(h*s),int(w*s)
        img_noisy_scaled = cv2.resize(imgNoisy, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        img_orig_scaled = cv2.resize(imgOrig, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        for i in range(0, new_h-patch_size+1, stride):
            for j in range(0, new_w-patch_size+1, stride):
                noisy_patch = img_noisy_scaled[i:i+patch_size, j:j+patch_size]
                orig_patch = img_orig_scaled[i:i+patch_size, j:j+patch_size]

                if augInc <= 1:
                    rdVariation = np.random.randint(0, max_data_aug_inc)
                    noisy_patches.append(DataAugm.DataVariation(noisy_patch, rdVariation))
                    orig_patches.append(DataAugm.DataVariation(orig_patch, rdVariation))
                else:
                    noisy_patches+=(DataAugm.DataAug(noisy_patch, augInc))
                    orig_patches+=(DataAugm.DataAug(orig_patch, augInc))
    
    return orig_patches,noisy_patches




'''
path = "C:/"
imgNoisy = cv2.imread(path + "0NoSP.jpeg",0)
imgOrig = cv2.imread(path + "exp.jpeg",0)

orig_patches,noisy_patches = GetPatches(imgNoisy,imgOrig,2,50,30)

print(len(orig_patches))
print(len(noisy_patches))

for i in range(len(orig_patches)):
    cv2.imwrite(path + "imp/" + str(i) + ".jpeg",noisy_patches[i])
    cv2.imwrite(path + "exp/" + str(i) + ".jpeg",orig_patches[i])
'''






