"""
SIMPLE MACHINE LEARNING
"""

# Most of the packages we'll use
# Pandas used for loading and cleaning data
# Numpy used for numerical purposes
# Scikit-learn is one of the most used machine learning packages 

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn import decomposition
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls

import seaborn as sns
sns.set()
"""
LOADING AND READING DATA
"""
train_data = np.load('../inputs/train.npy')
train_data = train_data.reshape(8134,12288)
train_labels = np.load('../inputs/train_target4.npy')
train_id = np.load('../inputs/train_id2.npy')
test_data = np.load('../inputs/test2.npy')
test_data = test_data.reshape(512, 12288)
test_id = np.load('../inputs/test_id.npy')

# Reshape feature array for SMOTE usage
#new_traindata = train_data.reshape(8134,12288)
# data class imbalance ratio after SMOTE usage (1055/1066/610; 1:1:2) 
target_df = pd.DataFrame(train_labels)
type_2 = target_df[target_df.values == 0].count()
type_3 = target_df[target_df.values == 1].count()
type_1 = target_df[target_df.values == 2].count()


x, y = train_data, train_labels
def plotClassificationData(x, y, title=""):
    palette = sns.color_palette()
    plt.scatter(x[y == 0, 0], x[y == 0, 1], label="normal", alpha=0.5,
                facecolor=palette[0], linewidth=0.15)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], label="benign", alpha=0.5,
                facecolor=palette[2], linewidth=0.15)
    plt.scatter(x[y == 1, 1], x[y == 1, 0], label="malignant", alpha=0.5,
                facecolor=palette[4], linewidth=0.15)
    plt.title(title)
    plt.legend()
    plt.show()

def count_classifieds(z): return sum(z)
def count_unclassifieds(z): return len(z) - sum(z)
def imbalance_ratio(z): return round(count_classifieds(z)/count_unclassifieds(z),1)
num_classified = count_classifieds(y)
num_unclassified = count_unclassifieds(y)
print("Number of classified clients: %s"%num_classified)
print("Number of unclassified clients: %s"%num_unclassified )
print("Imbalance ratio: %s"%imbalance_ratio(y))
pca = decomposition.PCA(n_components=3)
xv = pca.fit_transform(x)
plotClassificationData(xv,y)    

# Split data for better metrics evaluation
transformed_data, transformed_labels = shuffle(train_data, train_labels)
from sklearn.cross_validation import train_test_split
transformed_data_train, transformed_data_test, transformed_labels_train, transformed_labels_test = train_test_split(transformed_data, transformed_labels, test_size=0.5, random_state=0)   

### Reduce class imbalance
# Reshape feature array for SMOTE usage
sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(transformed_data_train, transformed_labels_train)
print train_labels.value_counts(), np.bincount(y_res)
transformed_data_train, transformed_labels_train = x_res, y_res
"""
MODEL TRAINING
"""
### RANDOMFOREST CLASSIFIER

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters for the fit
forest = RandomForestClassifier(n_estimators = 50)

# Fit the training data to the Language labels and create the decision trees
forest = forest.fit(transformed_data_train,transformed_labels_train)

# Take the same decision trees and run it on the test data for prediction
forest_preds = forest.predict_proba(transformed_data_test)

# Cross Validation
forest_scores = cross_val_score(forest, transformed_data_test, transformed_labels_test, cv = 5, verbose=2)
print('Prediction Accuracy in Percentage form:', forest_scores.mean() * 100)

# Confusion Matrix
print(__doc__)

class_names = np.array(['type 2', 'type 3', 'type 1'])

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(transformed_labels_test, forest_preds)
np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# More metrics for better assessment
print("\tPrecision: %1.3f" % precision_score(transformed_labels_test, forest_preds))
print("\tRecall: %1.3f" % recall_score(transformed_labels_test, forest_preds))
print("\tF1: %1.3f\n" % f1_score(transformed_labels_test, forest_preds))

### ADABOOST CLASSIFIER

from sklearn.ensemble import AdaBoostClassifier

# Create the AdaBoost object which will include all the parameters for the fit
adb = AdaBoostClassifier(n_estimators = 50)

# Fit the training data to the Language labels
adb = adb.fit(transformed_data_train,transformed_labels_train)

# Take the AdaBoost model and run it on the test data for prediction
adb_preds = adb.predict_proba(transformed_data_test)

# Cross Validation
adb_scores = cross_val_score(adb, transformed_data_test, transformed_labels_test, cv = 5, verbose=2)
print('Prediction Accuracy in Percentage form:', adb_scores.mean() * 100)

# Confusion Matrix
print(__doc__)

class_names = np.array(['type 2', 'type 3', 'type 1'])

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(transformed_labels_test, adb_preds)
np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# More metrics for better assessment
#print("\tPrecision: %1.3f" % precision_score(transformed_labels_test, adb_preds))
#print("\tRecall: %1.3f" % recall_score(transformed_labels_test, adb_preds))
#print("\tF1: %1.3f\n" % f1_score(transformed_labels_test, adb_preds))

### SVM CLASSIFIER

from sklearn.linear_model import SGDClassifier
svc = SGDClassifier(loss='hinge', n_iter=50, alpha=0.01)
svc = svc.fit(transformed_data_train,transformed_labels_train)
svc_preds = svc.predict(transformed_data_test)

# Cross Validation
svc_scores = cross_val_score(svc, transformed_data_test, transformed_labels_test, cv = 5, verbose=2)
print('Prediction Accuracy in Percentage form:', svc_scores.mean() * 100)

# Plot SVM hyperplanes
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#from sklearn.linear_model import SGDClassifier

# we create 40 separable points

# fit the model and get the separating hyperplane
X, y = transformed_data_train,transformed_labels_train

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y, verbose=1)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]

# get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y, verbose=1)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]

# plot separating hyperplanes and samples
h0 = plt.plot(xx, yy, 'k-', label='no weights')
h1 = plt.plot(xx, wyy, 'k--', label='with weights')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.legend()

plt.axis('tight')
plt.show()

# Confusion Matrix
print(__doc__)

class_names = np.array(['type 2', 'type 3', 'type 1'])

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(transformed_labels_test, svc_preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

### NAIVE BAYES CLASSIFIER

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(transformed_data_train, transformed_labels_train)
gnb_preds = gnb.predict(transformed_data_test)

# Cross_validation
gnb_scores = cross_val_score(gnb, transformed_data_test, transformed_labels_test, cv = 5, verbose=2)
print('Prediction Accuracy in Percentage form:', gnb_scores.mean() * 100)

# Confusion Matrix
print(__doc__)

class_names = np.array(['type 2', 'type 3', 'type 1'])

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(transformed_labels_test, gnb_preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# More metrics for better assessment
print("\tPrecision: %1.3f" % precision_score(transformed_labels_test, gnb_preds))
print("\tRecall: %1.3f" % recall_score(transformed_labels_test, gnb_preds))
print("\tF1: %1.3f\n" % f1_score(transformed_labels_test, gnb_preds))

""""""
SUBMISSION
""""""
preds = loaded_model.predict(test_data, batch_size=8, verbose=1)
df = pd.DataFrame(forest_preds, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = test_id
df.to_csv("submissions/"'submission_' + info_string + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv', index=False)
