import cv2
import os
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics

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
    
def disp_performance(clf,expected,predicted):
    print("Classification report for classifier %s:%s\n" % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    acc = accuracy_score(expected,predicted)
    print("\nClassification Accuracy Score: %s" % acc)
    
def classify_DecisionTree(features,labels):
    print("Training Decision Tree classifier...")
    n_samples = len(features)
    data = features.reshape((n_samples, -1))
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(data, labels)
    
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

dt_clf = classify_DecisionTree(Xtrain,Ytrain)

n_samples = len(Xtest)
expected = Ytest # for the sake of clarity

predicted_dt = dt_clf.predict(Xtest.reshape(n_samples,-1))

disp_performance(dt_clf,expected,predicted_dt)
