# For example, here's several helpful packages to load in 
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra                
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
				    
# Input data files are available in the "../input/" directory.        
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output                                                                             
print(check_output(["ls", "../inputs"]).decode("utf8"))
						    
# Any results you write to the current directory are saved as output.
				
np.random.seed(2017)                         
				  
import os                                                            
import glob                         
import cv2                                           
import datetime                                
import time                                  
import warnings              
import json
warnings.filterwarnings("ignore")                        
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, Model, model_from_json 
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, ELU, SReLU, LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop, Nadam, Adagrad, Adamax, TFOptimizer, Adadelta
from keras.regularizers import l2, activity_l2, l1, l1l2, activity_l1 
from keras.utils import np_utils

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard 
from keras import __version__ as keras_version             

"""
Reading and loading data
"""
"""
### REMOVING CORRUPTED IMAGES BY CONVERTING TO ARRAY THEN STORING INTO DIFFERENT FORMAT
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0,0]}]

def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (64, 64), cv2.INTER_LINEAR) #use cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    return [path, resized]

def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata

train = glob.glob('inputs/train/**/*.jpg') + glob.glob('inputs/additional/**/*.jpg')
train = pd.DataFrame([[p.split('/')[3],p.split('/')[3],p] for p in train], columns = ['type','image','path']) #[::5] limit for Kaggle Demo
train = im_stats(train)
train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
train_data = normalize_image_features(train['path'])
np.save('train.npy', train_data, allow_pickle=True, fix_imports=True)

le = LabelEncoder()
train_target = le.fit_transform(train['type'].values)
print(le.classes_) #in case not 1 to 3 order
np.save('train_target.npy', train_target, allow_pickle=True, fix_imports=True)

test = glob.glob('inputs/test/*.jpg')
test = pd.DataFrame([[p.split('/')[2],p] for p in test], columns = ['image','path']) #[::20] #limit for Kaggle Demo
test_data = normalize_image_features(test['path'])
np.save('test.npy', test_data, allow_pickle=True, fix_imports=True)

train_id = train.image.values
np.save('train_id.npy', train_id, allow_pickle=True, fix_imports=True)

test_id = test.image.values
np.save('test_id.npy', test_id, allow_pickle=True, fix_imports=True)
"""
"""
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_dim_ordering('tf')
K.set_floatx('float32')

import pandas as pd
import numpy as np
np.random.seed(17)

def create_model(opt_='adamax'):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf', input_shape=(3, 64, 64))) #use input_shape=(3, 64, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_, loss='categorical_crossentropy', metrics=['accuracy']) 
    return model

datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
datagen.fit(train_data)

model = create_model()
model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=200, samples_per_epoch=len(x_train), verbose=20, validation_data=(x_val_train, y_val_train))


pred = model.predict_proba(test_data)
df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = test_id
df.to_csv('submission.csv', index=False)
"""


train_data = np.load('inputs/train.npy')
train_target = np.load('inputs/train_target4.npy')
train_id = np.load('inputs/train_id2.npy')
test_data = np.load('inputs/test2.npy')
test_id = np.load('inputs/test_id.npy')
train_target = np_utils.to_categorical(train_target, 3)

train_data = shuffle(train_data)
train_target = shuffle(train_target)
test_data = shuffle(test_data)

# Split data for model
#train_data,train_data_val,train_target,train_target_val = train_test_split(train_data,train_target,test_size=0.3, random_state=17)

early_stopping = EarlyStopping(monitor='val_loss', patience=50)
now = datetime.datetime.now()
checkpointer = ModelCheckpoint(filepath="weights/"'checkpointer_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.hdf5', verbose=1, save_best_only=True)
batch_size = 16
nb_epoch = 1000

### image dimensions
num_channels = 3
imageSize = (64, 64)
img_width, img_height = imageSize[0], imageSize[1]
    
"""
THE MODEL
"""
def build_model():
    wr1 = 1e-07
    wr2 = 1e-06
    wr3 = 1e-05
    wr4 = 1e-06
    wr5 = 1e-05

    activation = 'relu'
    optimizer = 'sgd'
     
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(num_channels, img_width, img_height), dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr1), activity_regularizer=activity_l1(wr1), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr1), activity_regularizer=activity_l1(wr1), init='lecun_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.85))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr2), activity_regularizer=activity_l1(wr2), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr2), activity_regularizer=activity_l1(wr2), init='lecun_uniform'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.85))

    """
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr3), activity_regularizer=activity_l1(wr3), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr3), activity_regularizer=activity_l1(wr3), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr3), activity_regularizer=activity_l1(wr3), init='lecun_uniform'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.85))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr4), activity_regularizer=activity_l1(wr4), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr4), activity_regularizer=activity_l1(wr4), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr4), activity_regularizer=activity_l1(wr4), init='lecun_uniform'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.95))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr5), activity_regularizer=activity_l1(wr5), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr5), activity_regularizer=activity_l1(wr5), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr5), activity_regularizer=activity_l1(wr5), init='lecun_uniform'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.95))
    """
    
    model.add(Flatten())

    model.add(Dense(512, activation=activation, W_regularizer=l2(wr4), init='lecun_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=activation, W_regularizer=l2(wr4), init='lecun_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation=activation, W_regularizer=l2(wr5), init='lecun_uniform'))
    model.add(Dropout(0.5))
    #model.add(Dense(128, activation=activation, W_regularizer=l2(wr5), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    
    model.add(Dense(3, activation='softmax', W_regularizer=l2(wr1), init='lecun_uniform'))

    sgd = SGD(lr=1e-1, momentum=0.9, clipvalue=0.5)
    adam = Adam(lr=1e-3)
    nadam = Nadam(lr=1e-2)
    adagrad = Adagrad(lr=1.0, decay=1e-2)
    adamax = Adamax(lr=1e-2)
    adadelta = Adadelta(lr=10)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    #early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    return model

def run_train(n_folds=5):
    num_fold = 5
    sum_score = 0
    models = []   
    callbacks = [
        early_stopping
    ]
    
    ### if we just want to train a single model without cross-validation, set n_folds to 0 or None
    if not n_folds:
        model = build_model()
        
        #X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=val_split)
        print('Training...')
        #print('Size of train split: ', len(X_train), len(Y_train))
        #print('Size of validation split: ', len(X_valid), len(Y_valid))

        h = model.fit(train_data, train_target, 
	    batch_size=batch_size,
	    #nb_epoch=nb_epoch,
	    verbose=1, 
	    validation_split=0.5, 
	    shuffle=True,
	    callbacks=[early_stopping, checkpointer])


        predictions_valid = model.predict(test_data, batch_size=batch_size, verbose=1)
        score = h.history['loss'][-1]
        print('Loss: ', score)
        sum_score += score
        models.append(model)
                     
    else:
        kf = KFold(len(train_id), n_folds=n_folds, shuffle=True, random_state=7)

        for train_index, test_index in kf:
            model = build_model()
            X_train = train_data[train_index]
            Y_train = train_target[train_index]
            X_valid = train_data[test_index]
            Y_valid = train_target[test_index]

            num_fold += 1
            print('Training on fold {} of {}...'.format(num_fold, n_folds))
            print('Size of train split: ', len(X_train), len(Y_train))
            print('Size of validation split: ', len(X_valid), len(Y_valid))

	    model = KerasClassifier(build_fn=build_model, nb_epoch=1000, batch_size=16, verbose=1)

	    h = model.fit(train_data, train_target, 
		batch_size=batch_size,
		#nb_epoch=nb_epoch,
		verbose=1, 
		validation_split=0.5, 
		shuffle=True,
		#class_weight=class_weight)
		callbacks=[early_stopping, checkpointer])

            predictions_valid = model.predict(test_data, batch_size=batch_size, verbose=1)
            score = h.history['loss'][-1]
            print('Loss for fold {0}: '.format(num_fold),score)
            sum_score += score*len(test_index)
            models.append(model)
        score = sum_score/len(train_data)
        
    print("Average loss across folds: ", score)
    
    info_string = "loss-{0:.2f}_{1}fold_{2}x{3}_{4}epoch_patience".format(score, n_folds, img_width, img_height, nb_epoch)
    return info_string, models

print(model)

# RMSE
np.sqrt(0.000156) * 48 

""""""
SAVING AND LOADING WEIGHTS
""""""
"""
# loading weights
###weights = open('net-specialists2.pickle', 'rb')

### saving the model and its weights
# serialize model to JSON
model_json = model.to_json()
with open('model_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('weights_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.h5')
print("Saved model to disk")

# load the previous weights' networks
f = h5py.File('model_relu_500epochs.h5')
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
	# we don't look at the last (fully-connected) layers in the savefile
	break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()

# load json and create model
json_file = open('model10.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_relu_220epochs100patients.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

preds = loaded_model.predict(test_data, batch_size=8, verbose=1)
    
# compile the model with a SGD/momentum optimizer
#model.compile(loss='categorical_crossentropy',
#	      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9))

"""
""""""
VALIDATION & SUBMISSION STAGE
""""""
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['Type_1', 'Type_2', 'Type_3'])
    #result2 = pd.DataFrame(predictions)
    result1.loc[:, 'image_name'] = pd.Series(test_id, index=result1.index)
    #result2.loc[:, 'image'] = pd.Series(test_id, index=result2.index)
    now = datetime.datetime.now()
    sub_file = "submissions/"'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def ensemble_predict(info_string, models):
    num_fold = 0
    yfull_test = []
    test_id = []
    n_folds = len(models)

    for i in range(n_folds):
        model = models[i]
        num_fold += 1
        print('Predicting on fold {} of {}'.format(num_fold, n_folds))
        #test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    preds = merge_several_folds_mean(yfull_test, n_folds)
    create_submission(preds, test_id, info_string)
    model_json = model.to_json()
    with open('model_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.json', "w") as json_file:
	json_file.write(model_json)

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 5
    info_string, models = run_train()
    #info_string, models = run_cross_validation_create_models(num_folds)
    ensemble_predict(info_string, models)
    