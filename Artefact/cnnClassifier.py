import keras
import cv2
import os
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_dataset(dataset_path, classes):
    images = []
    labels = []
    img_names = []
    cls = [] #classes
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(dataset_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            # Reading the image using OpenCV
            bgr_img = cv2.imread(fl)
            b,g,r = cv2.split(bgr_img)       # get b,g,r
            
            rgb_img = cv2.merge([r,g,b])     # switch it to rgb    
            rgb_img = rgb_img.astype(np.float32)
            rgb_img = np.multiply(rgb_img, 1.0 / 255.0)    
            
            gray_img = rgb2gray(rgb_img)
            images.append(gray_img)
            labels.append(index)
            
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

def reshape_Features(features,img_rows,img_cols,shapeFlag):
    if K.image_data_format() == 'channels_first':
        features = features.reshape(features.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        features = features.reshape(features.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    if shapeFlag == 1:
        return features,input_shape
    else:
        return features
    
def disp_performance_cnn(expected,predicted):
    print("\nClassification report for CNN softmax classifier:\n")
    report = classification_report(expected, predicted)
    print(report)
    acc = accuracy_score(expected,predicted)
    print("\nClassification Accuracy Score: %s" % acc)

def classify_CNN(x_train,y_train,x_valid,y_valid,batch_size,epochs,num_classes):
    print("Training CNN...")
    img_rows = 28
    img_cols = 28
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_valid.shape[0], 'validation samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    clf = Sequential()
    clf.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    clf.add(Conv2D(64, (3, 3), activation='relu'))
    clf.add(MaxPooling2D(pool_size=(2, 2)))
    clf.add(Dropout(0.25))
    clf.add(Flatten())
    clf.add(Dense(128, activation='relu'))
    clf.add(Dropout(0.5))
    clf.add(Dense(num_classes, activation='softmax'))
    
    clf.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    clf.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid))
    
    return clf
    
################################################################################
######################## DATA ACQUISITION ######################################
################################################################################
    
classes = ['0','1','2','3','4','5','6','7','8','9']
train_in = ('.\dataset\\training\\')
valid_in = ('.\dataset\\validation\\')
test_in = ('.\dataset\\test\\')

# load input datasets. X___ = image features, Y___ = image labels. Store image names for the shuffle function.
print("Loading training dataset...\n")
Xtrain, Ytrain, train_img_names, train_cls = load_dataset(train_in, classes)
Xtrain, Ytrain, train_img_names, train_cls = shuffle(Xtrain, Ytrain, train_img_names, train_cls)  

print("\nLoading validation dataset...\n")
Xvalid, Yvalid, valid_img_names, valid_cls = load_dataset(valid_in, classes)
Xvalid, Yvalid, valid_img_names, valid_cls = shuffle(Xvalid, Yvalid, valid_img_names, valid_cls)

print("\nLoading test dataset...\n")
Xtest, Ytest, test_img_names, test_cls = load_dataset(test_in, classes)
Xtest, Ytest, test_img_names, test_cls = shuffle(Xtest, Ytest, test_img_names, test_cls)

print("\nLoading Complete!\n")

###############################################################################
####################### CLASSIFICATION PHASE ##################################
###############################################################################
epochs = 12
img_rows = 28
img_cols = 28
batch_size = 128
num_classes = len(classes)

cnn_clf = classify_CNN(Xtrain,Ytrain,Xvalid,Yvalid,batch_size,epochs,num_classes)

n_samples = len(Xtest)

Xtest = reshape_Features(Xtest,img_rows,img_cols,0)
Ytest = keras.utils.to_categorical(Ytest, num_classes)

predicted_CNN = cnn_clf.predict_classes(Xtest)
predicted_CNN = keras.utils.to_categorical(predicted_CNN, num_classes)

score = cnn_clf.evaluate(Xtest, Ytest)

disp_performance_cnn(Ytest,predicted_CNN)