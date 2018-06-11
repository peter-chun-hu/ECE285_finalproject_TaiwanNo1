import numpy as np
import pandas as pd
import glob
from PIL import Image
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

SIZE = 128

# load file
train_images = glob.glob("whaleIdentification/train/*jpg")
test_images = glob.glob("whaleIdentification/test/*jpg")
df = pd.read_csv("whaleIdentification/train.csv")

df["Image"] = df["Image"].map( lambda x : "whaleIdentification/train/"+x)
ImageToLabelDict = dict( zip( df["Image"], df["Id"]))

#image are imported with a resizing and a black and white conversion
def ImportImage(filename):
    img = Image.open(filename).convert("RGB").resize((SIZE,SIZE)) #.convert("LA")
    return np.array(img)[:,:,:]

# transfer label to one-hot
class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform(x)
        return self.ohe.fit_transform(features.reshape(-1,1))
    def transform(self, x):
        return self.ohe.transform(self.la.transform(x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform(self.ohe.inverse_tranform(x))
    def inverse_labels(self, x):
        return self.le.inverse_transform(x)

# transfer label to one-hot
y = list(map(ImageToLabelDict.get, train_images))
lohe = LabelOneHotEncoder()
y_cat = lohe.fit_transform(y)
print (y_cat.shape)

# constructing class weights
# due to imbalanced dataset
WeightFunction = lambda x : 1./x**0.75
ClassLabel2Index = lambda x : lohe.le.inverse_tranform([[x]])
CountDict = dict(df["Id"].value_counts())
class_weight_dic = {lohe.le.transform([image_name])[0] : WeightFunction(count) for image_name, count in CountDict.items()}
del CountDict

#use of an image generator for preprocessing and data augmentation
x = x.reshape((-1,SIZE,SIZE,3))
input_shape = x[0].shape
x_train = x.astype("float32")
y_train = y_cat

image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rescale=1./255,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True)

#training the image preprocessing
image_gen.fit(x_train, augment=True)

# transfer learning requries RGB
x_train_3 = x.reshape((-1,SIZE,SIZE,3))
x_train_3 = x_train_3.astype("float32")


# ========== transfer learning ==========

# building model
batch_size = 128
num_classes = len(y_cat.toarray()[0])
epochs = 100

# model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (SIZE, SIZE, 3))

model = applications.InceptionResNetV2(weights = None, include_top=False, input_shape = (SIZE, SIZE, 3))

# Freeze the layers which you don't want to train. Here I am freezing all layers due to small dataset.
for layer in model.layers:
    layer.trainable = False
#Now we will be training only the classifiers (FC layers)

#Adding custom Layers 
x_model = model.output
x_model = Flatten()(x_model)
x_model = Dense(2048, activation="relu")(x_model)
x_model = Dropout(0.5)(x_model)
x_model = Dense(2048, activation="relu")(x_model)
predictions = Dense(4251, activation="softmax")(x_model)

# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)

# load previous training model
model_final.load_weights("IncepResV2_3.h5")

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adadelta(), metrics=["accuracy"])

#training the image preprocessing
train_datagen = image_gen.fit(x_train_3, augment=True)

# Save the model according to the conditions  
# checkpoint = ModelCheckpoint("vgg19_1.h5", monitor='val_acc', verbose=1, save_weights_only=False, mode='auto', period=1) #, save_best_only=True
checkpoint = ModelCheckpoint("IncepResV2_3.h5", monitor='val_acc', verbose=1, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model 
with tf.device('/gpu:0'):
    model_final.fit_generator(
    image_gen.flow(x_train_3, y_train.toarray(), batch_size=batch_size),
    steps_per_epoch = x_train_3.shape[0]//batch_size,
    epochs = epochs,
    callbacks = [checkpoint, early])


# total training accuracy 
score_3 = model_final.evaluate(x_train_3, y_train, verbose=0)
print('Training loss: {0:.4f}\nTraining accuracy:  {1:.4f}'.format(*score_3))

# output of testing images (prediction) 
import warnings
from os.path import split

with open("transfer_submission.csv","w") as f:
    with warnings.catch_warnings():
        f.write("Image,Id\n")
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        for image in test_images:
            img = ImportImage( image)
            x = img.astype( "float32")
            #applying preprocessing to test images
            x = image_gen.standardize( x.reshape(1,SIZE,SIZE,3))
            y = model_final.predict_proba(x.reshape(1,SIZE,SIZE,3))
            predicted_args = np.argsort(y)[0][::-1][:5]
            predicted_tags = lohe.inverse_labels( predicted_args)
            image = split(image)[-1]
            predicted_tags = " ".join( predicted_tags)
            f.write("%s,%s\n" %(image, predicted_tags))
