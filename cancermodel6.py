"""""" 
CANCER DIAGNOSTICS MODEL
""""""
import numpy as np
import pandas as pd
import glob
import time
import datetime

from PIL import ImageFilter, ImageStat, Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import cv2

from multiprocessing import Pool, cpu_count

from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, classification_report, confusion_matrix, f1_score
from sklearn.utils import class_weight

from imblearn.over_sampling import SMOTE

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Nadam, Adagrad, Adamax, TFOptimizer, Adadelta, TFOptimizer
from keras import optimizers
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard 
from keras.regularizers import l2, activity_l2, l1, l1l2, activity_l1, activity_l1l2 

from keras import __version__ as keras_version             


""""""
Loading and reading data
""""""
"""
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


    

train = glob.glob('../inputs/train/**/*.jpg') + glob.glob('../inputs/additional/**/*.jpg')
train = pd.DataFrame([[p.split('/')[3],p.split('/')[3],p] for p in train], columns = ['type','image','path']) #[::5] limit for Kaggle Demo
train = im_stats(train)
split = train['path'].apply(lambda x: pd.Series(str(x).split('/')[3]))
train['y'] = split

## Removing and replacing data files
remove_img = pd.read_csv('../removed_files.csv')
replace_Labels = pd.read_csv('../fixed_labels_v2.csv')

# remove given bad images from 'removed_files.csv'
print(train[~train.image.isin(remove_img.filename)])   
new_df = train[~train.image.isin(remove_img.filename)]    
#newlabels = replace_Labels.rename(columns={'filename': 'image'})

# Replacing labels for less class imbalance
# set matching columns to multi-level index
x1 = new_df.set_index(['image'])['y']
x2 = replace_Labels.set_index(['filename'])['new_label']
# call update function, this is inplace
x1.update(x2)
# replace the values in original df1
new_df['y'] = x1.values

#remove bad images
train = new_df[new_df['size'] != '0 0'].reset_index(drop=True) 
train_data = normalize_image_features(train['path'])
np.save('inputs/train.npy', train_data, allow_pickle=True, fix_imports=True)

#le = LabelEncoder()
#train_target = le.fit_transform(train['type'].values)
#print(le.classes_) #in case not 1 to 3 order

y_vector = train['y']
y_list = []
for row in y_vector:
    if (row == 'Type_1'):
	row = 0
    elif (row == 'Type_2'):
	row = 1
    else:
	row = 2
    y_list.append(row)

train_target = np.array(y_list, dtype=np.uint8)
np.save('inputs/train_target4.npy', train_target, allow_pickle=True, fix_imports=True)

test = glob.glob('inputs/test/*.jpg')
test = pd.DataFrame([[p.split('/')[2],p] for p in test], columns = ['image','path']) #[::20] #limit for Kaggle Demo
test_data = normalize_image_features(test['path'])
np.save('test2.npy', test_data, allow_pickle=True, fix_imports=True)

train_id = train.image.values
np.save('train_id2.npy', train_id, allow_pickle=True, fix_imports=True)

test_id = test.image.values
np.save('test_id.npy2', test_id, allow_pickle=True, fix_imports=True)
"""

train_data = np.load('../inputs/train.npy')
train_labels = np.load('../inputs/train_target4.npy')
train_id = np.load('../inputs/train_id2.npy')
test_data = np.load('../inputs/test2.npy')
test_id = np.load('../inputs/test_id.npy')
train_target = np_utils.to_categorical(train_labels, 3)

train_data = shuffle(train_data)
train_target = shuffle(train_target)
test_data = shuffle(test_data)


'''
### Reduce class imbalance
# Reshape feature array for SMOTE usage
new_traindata = train_data.reshape(8134,12288)
sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(new_traindata, train_labels)
print train_target.value_counts(), np.bincount(y_res)

np.save('../inputs/new_traindata.npy', x_res, allow_pickle=True, fix_imports=True)
np.save('../inputs/new_trainlabels.npy', y_res, allow_pickle=True, fix_imports=True)
'''

""""""
VISUALS
""""""
"""
train = glob.glob('inputs/train/**/*.jpg') + glob.glob('inputs/additional/**/*.jpg')
train = pd.DataFrame([[p.split('/')[3],p.split('/')[3],p] for p in train], columns = ['type','image','path'])

test = glob.glob('inputs/test/*.jpg')
test = pd.DataFrame([[p.split('/')[2],p] for p in test], columns = ['image','path'])

sub = pd.read_csv('inputs/sample_submission.csv')
print(len(train),len(test),len(sub))

# bar charts of different cancer types
types = train.groupby('type', as_index=False)['path'].count()
_ = types.plot(kind='bar', x='type', y='path', figsize=(7,4))


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

# bar chart of different sizes of tumors
train = im_stats(train)
sizes = train.groupby('size', as_index=False)['path'].count()
_ = sizes.plot(kind='bar', x='size', y='path', figsize=(7,4))
"""

""""""
Model
""""""
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
now = datetime.datetime.now()
checkpointer = ModelCheckpoint(filepath="../weights/"'checkpointer_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.h5', verbose=1, save_best_only=True)
#class_weight = class_weight.compute_class_weight('balanced', np.unique(train_target), train_target)

### image dimensions
num_channels = 3
imageSize = (64, 64)
img_width, img_height = imageSize[0], imageSize[1]

# loading weights for batching of training procedure
weights_path = '../weights/checkpointer__2017-05-16-16-36.h5'

# Split data for model
new_train_data,train_data_val,new_train_target,train_target_val = train_test_split(train_data,train_target,test_size=0.5, random_state=17)

def create_model():
    wr1 = 1e-05
    wr2 = 1e-04
    wr3 = 1e-04
    wr4 = 1e-04
    wr5 = 1e-03

    activation = 'relu'
    optimizer = 'sgd'
     
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(num_channels, img_width, img_height), dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1l2(wr1), activity_regularizer=activity_l2(wr1), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1l2(wr1), activity_regularizer=activity_l2(wr1), init='lecun_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.85))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1l2(wr2), activity_regularizer=activity_l2(wr2), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1l2(wr2), activity_regularizer=activity_l2(wr2), init='lecun_uniform'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.85))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr3), activity_regularizer=activity_l1(wr3), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr3), activity_regularizer=activity_l1(wr3), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr3), activity_regularizer=activity_l1(wr3), init='lecun_uniform'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.85))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr4), activity_regularizer=activity_l1(wr4), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr4), activity_regularizer=activity_l1(wr4), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation=activation, dim_ordering='tf', W_regularizer=l1(wr4), activity_regularizer=activity_l1(wr4), init='lecun_uniform'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.95))

    """
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

    model.add(Dense(128, activation=activation, W_regularizer=l2(wr4), init='lecun_uniform'))
    model.add(Dropout(0.85))
    model.add(Dense(128, activation=activation, W_regularizer=l2(wr4), init='lecun_uniform'))
    model.add(Dropout(0.75))
    #model.add(Dense(128, activation=activation, W_regularizer=l2(wr5), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    #model.add(Dense(128, activation=activation, W_regularizer=l2(wr5), init='lecun_uniform'))
    #model.add(Dropout(0.5))
  
    model.add(Dense(3, activation='softmax'))

    sgd = SGD(lr=1e-4, momentum=0.9, clipvalue=0.5, nesterov=True)
    adam = Adam(lr=1e-4, decay=1e-6)
    nadam = Nadam(lr=1e-2)
    adagrad = Adagrad(lr=1e-2)
    adamax = Adamax(lr=1e-1)
    adadelta = Adadelta(lr=10)

    model.compile(optimizer=adam, loss='squared_hinge', metrics=['mse'])
    
    model.summary()
   
    #model.load_weights(weights_path)

    nb_epoch = 500
    #model = KerasClassifier(build_fn=create_model, nb_epoch=nb_epoch, batch_size=64, verbose=1)

    model_result = model.fit(new_train_data, new_train_target, 
			 nb_epoch=nb_epoch, 
			 validation_data=(train_data_val, train_target_val), 
			 shuffle=True, 
			 batch_size=64,
			 class_weight = 'auto',
			 callbacks=[early_stopping, checkpointer], verbose=1)
    pred = model.predict_proba(test_data)
    score = model_result.history['val_loss'][-1]
    # Save model 
    info_string = "loss-{0:.2f}_{0}fold_{1}x{2}_{3}epoch_patience".format(score, img_width, img_height, nb_epoch)
    model_json = model.to_json()
    with open("../models/"'model_' + info_string + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.json', "w") as json_file:
        json_file.write(model_json)

    return model_result, pred, info_string

model_result, pred, info_string = create_model()
#pred = model_result.predict(test_data)
print(pred)

## PLOT ACCURACY & LOSS
# list all data in history
print(model_result.history.keys())
# summarize history for accuracy
plt.plot(model_result.history['mean_squared_error'])
plt.plot(model_result.history['val_mean_squared_error'])
plt.title('model MSE')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_result.history['loss'])
plt.plot(model_result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Confusion Matrix & F1-score
y_pred = model_result.model.predict_classes(train_data_val)
target_names = ['type_1', 'type_2', 'type_3']
print(classification_report(np.argmax(train_target_val, axis=1), y_pred, target_names=target_names))
print(confusion_matrix(np.argmax(train_target_val, axis=1), y_pred))

#model = KerasClassifier(build_fn=create_model, nb_epoch=200, batch_size=16, verbose=1)
#opts_ = ['adamax'] #['adadelta','sgd','adagrad','adam','adamax']
#epochs = np.array([10])
#batches = np.array([10])
#param_grid = dict(nb_epoch=epochs, batch_size=batches, opt_=opts_)
#grid = GridSearchCV(estimator=model, cv=StratifiedKFold(n_splits=2), param_grid=param_grid, verbose=20)
#grid_result = grid.fit(train_data, train_target)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
#results = cross_val_score(model, train_data, train_target, cv=kfold)
#model_result = model.fit(train_data, train_target, validation_split=0.1, shuffle=True, callbacks=[early_stopping, checkpointer], verbose=1)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

#p = model_result.model.predict_proba(test_data)
df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = test_id
df.to_csv("../submissions/"'submission_' + '_' + info_string + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv', index=False)

""""""
SAVING AND LOADING WEIGHTS
""""""
"""
# loading weights
###weights = open('net-specialists2.pickle', 'rb')

### saving the model and its weights
# serialize model to JSON
model_json = model_result.to_json()
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

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# compile the model with a SGD/momentum optimizer
#model.compile(loss='categorical_crossentropy',
#	      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9))
"""
""""""
SUBMISSION
""""""

# load json and create model
json_file = open('models/model_loss-1.01_1.00573947689fold_64x64_10epoch_patience_2017-05-11-12-30.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights/checkpointer__2017-05-11-12-30.h5")
print("Loaded model from disk")

preds = model_result.model.predict_proba(test_data, batch_size=8, verbose=1)
df = pd.DataFrame(preds, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = test_id
df.to_csv("../submissions/"'submission_' + info_string + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv', index=False)

1055/1066/610