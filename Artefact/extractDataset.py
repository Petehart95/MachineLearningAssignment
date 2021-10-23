from PIL import Image
import os
        
inputDir1 = ('.\digits-train-5000')
inputDir2 = ('.\digits-validation-1000')
inputDir3 = ('.\digits-test-500')

outTrainDir0 = ('.\dataset\\training\\0')
outTrainDir1 = ('.\dataset\\training\\1')
outTrainDir2 = ('.\dataset\\training\\2')
outTrainDir3 = ('.\dataset\\training\\3')
outTrainDir4 = ('.\dataset\\training\\4')
outTrainDir5 = ('.\dataset\\training\\5')
outTrainDir6 = ('.\dataset\\training\\6')
outTrainDir7 = ('.\dataset\\training\\7')
outTrainDir8 = ('.\dataset\\training\\8')
outTrainDir9 = ('.\dataset\\training\\9')

outValidDir0 = ('.\dataset\\validation\\0')
outValidDir1 = ('.\dataset\\validation\\1')
outValidDir2 = ('.\dataset\\validation\\2')
outValidDir3 = ('.\dataset\\validation\\3')
outValidDir4 = ('.\dataset\\validation\\4')
outValidDir5 = ('.\dataset\\validation\\5')
outValidDir6 = ('.\dataset\\validation\\6')
outValidDir7 = ('.\dataset\\validation\\7')
outValidDir8 = ('.\dataset\\validation\\8')
outValidDir9 = ('.\dataset\\validation\\9')

outTestDir0 = ('.\dataset\\test\\0')
outTestDir1 = ('.\dataset\\test\\1')
outTestDir2 = ('.\dataset\\test\\2')
outTestDir3 = ('.\dataset\\test\\3')
outTestDir4 = ('.\dataset\\test\\4')
outTestDir5 = ('.\dataset\\test\\5')
outTestDir6 = ('.\dataset\\test\\6')
outTestDir7 = ('.\dataset\\test\\7')
outTestDir8 = ('.\dataset\\test\\8')
outTestDir9 = ('.\dataset\\test\\9')

if not os.path.exists(outTrainDir0):
    os.makedirs(outTrainDir0)
if not os.path.exists(outTrainDir1):
    os.makedirs(outTrainDir1)
if not os.path.exists(outTrainDir2):
    os.makedirs(outTrainDir2)
if not os.path.exists(outTrainDir3):
    os.makedirs(outTrainDir3)
if not os.path.exists(outTrainDir4):
    os.makedirs(outTrainDir4)
if not os.path.exists(outTrainDir5):
    os.makedirs(outTrainDir5)
if not os.path.exists(outTrainDir6):
    os.makedirs(outTrainDir6)
if not os.path.exists(outTrainDir7):
    os.makedirs(outTrainDir7)
if not os.path.exists(outTrainDir8):
    os.makedirs(outTrainDir8)
if not os.path.exists(outTrainDir9):
    os.makedirs(outTrainDir9)
    
if not os.path.exists(outValidDir0):
    os.makedirs(outValidDir0)
if not os.path.exists(outValidDir1):
    os.makedirs(outValidDir1)
if not os.path.exists(outValidDir2):
    os.makedirs(outValidDir2)
if not os.path.exists(outValidDir3):
    os.makedirs(outValidDir3)
if not os.path.exists(outValidDir4):
    os.makedirs(outValidDir4)
if not os.path.exists(outValidDir5):
    os.makedirs(outValidDir5)
if not os.path.exists(outValidDir6):
    os.makedirs(outValidDir6)
if not os.path.exists(outValidDir7):
    os.makedirs(outValidDir7)
if not os.path.exists(outValidDir8):
    os.makedirs(outValidDir8)
if not os.path.exists(outValidDir9):
    os.makedirs(outValidDir9)
    
if not os.path.exists(outTestDir0):
    os.makedirs(outTestDir0)
if not os.path.exists(outTestDir1):
    os.makedirs(outTestDir1)
if not os.path.exists(outTestDir2):
    os.makedirs(outTestDir2)
if not os.path.exists(outTestDir3):
    os.makedirs(outTestDir3)
if not os.path.exists(outTestDir4):
    os.makedirs(outTestDir4)
if not os.path.exists(outTestDir5):
    os.makedirs(outTestDir5)
if not os.path.exists(outTestDir6):
    os.makedirs(outTestDir6)
if not os.path.exists(outTestDir7):
    os.makedirs(outTestDir7)
if not os.path.exists(outTestDir8):
    os.makedirs(outTestDir8)
if not os.path.exists(outTestDir9):
    os.makedirs(outTestDir9)
    
    
for file in os.listdir(inputDir1):
    fileName = os.fsdecode(file)
    selectedNumber = int(fileName[0])
    selectedFilePath = inputDir1 + '\\' + fileName
    im = Image.open(selectedFilePath)
    if (selectedNumber == 0):
        newFilePath = outTrainDir0 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 1):
        newFilePath = outTrainDir1 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 2):
        newFilePath = outTrainDir2 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 3):
        newFilePath = outTrainDir3 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 4):
        newFilePath = outTrainDir4 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 5):
        newFilePath = outTrainDir5 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 6):
        newFilePath = outTrainDir6 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 7):
        newFilePath = outTrainDir7 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 8):
        newFilePath = outTrainDir8 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 9):
        newFilePath = outTrainDir9 + '\\' + fileName
        im.save(newFilePath)
    else:
        continue    

for file in os.listdir(inputDir2):
    fileName = os.fsdecode(file)
    selectedNumber = int(fileName[0])
    selectedFilePath = inputDir2 + '\\' + fileName
    im = Image.open(selectedFilePath)
    if (selectedNumber == 0):
        newFilePath = outValidDir0 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 1):
        newFilePath = outValidDir1 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 2):
        newFilePath = outValidDir2 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 3):
        newFilePath = outValidDir3 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 4):
        newFilePath = outValidDir4 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 5):
        newFilePath = outValidDir5 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 6):
        newFilePath = outValidDir6 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 7):
        newFilePath = outValidDir7 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 8):
        newFilePath = outValidDir8 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 9):
        newFilePath = outValidDir9 + '\\' + fileName
        im.save(newFilePath)
    else:
        continue    
    
for file in os.listdir(inputDir3):
    fileName = os.fsdecode(file)
    selectedNumber = int(fileName[0])
    selectedFilePath = inputDir3 + '\\' + fileName
    im = Image.open(selectedFilePath)
    if (selectedNumber == 0):
        newFilePath = outTestDir0 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 1):
        newFilePath = outTestDir1 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 2):
        newFilePath = outTestDir2 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 3):
        newFilePath = outTestDir3 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 4):
        newFilePath = outTestDir4 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 5):
        newFilePath = outTestDir5 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 6):
        newFilePath = outTestDir6 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 7):
        newFilePath = outTestDir7 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 8):
        newFilePath = outTestDir8 + '\\' + fileName
        im.save(newFilePath)
    elif (selectedNumber == 9):
        newFilePath = outTestDir9 + '\\' + fileName
        im.save(newFilePath)
    else:
        continue    