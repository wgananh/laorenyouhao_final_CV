# -*- coding: utf-8 -*-
'''
训练情感分析模型
'''

# import the necessary packages
from A_Final_Sys.oldcare.datasets import SimpleDatasetLoader
from A_Final_Sys.oldcare.preprocessing import AspectAwarePreprocessor
from A_Final_Sys.oldcare.preprocessing import ImageToArrayPreprocessor
from A_Final_Sys.oldcare.conv import MiniVGGNet
from A_Final_Sys.oldcare.conv import MiniGOOGLENet
from A_Final_Sys.oldcare.callbacks import TrainingMonitor
from imutils import paths
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# 全局变量
pre_model_name = 'miniGOOGLE_Adam_50_7'
model_name = 'miniGOOGLE_Adam_20_7_en'
dataset_path = 'C:/Users/whg/Downloads/fer'
pre_model_path = 'models/' + pre_model_name + '.hdf5'
output_model_path = 'models/' + model_name + '.hdf5'
output_plot_path = 'plots/' + model_name + '.png'

# 全局常量
TARGET_WIDTH = 28
TARGET_HEIGHT = 28
BATCH_SIZE = 64
EPOCHS = 20
CLASS = 7
LR_INIT = 0.01
DECAY = LR_INIT / 15
MOMENTUM = 0.9

# 加载图片
aap = AspectAwarePreprocessor(TARGET_WIDTH, TARGET_HEIGHT)
iap = ImageToArrayPreprocessor()

print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, 500, True)
data = data.astype("float") / 255.0

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), CLASS)
print(le.classes_)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.20, stratify=labels, random_state=42)


aug = ImageDataGenerator(rotation_range=45, width_shift_range=0.15,
                         height_shift_range=0.15, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True,
                         fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")

# miniVGG模型
# model = MiniVGGNet.build(width=TARGET_WIDTH,
#                          height=TARGET_HEIGHT, depth=1, classes=7)

# miniGOOGLE模型
model = MiniGOOGLENet.build(width=TARGET_WIDTH,
                            height=TARGET_HEIGHT, depth=1, classes=CLASS)
# model = load_model(pre_model_path)

# opt = SGD(lr=LR_INIT, decay=DECAY, momentum=MOMENTUM, nesterov=True)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# construct the set of callbacks
callbacks = [TrainingMonitor(output_plot_path)]

# train the network
print("[INFO] training network...")
# H = model.fit(trainX, trainY, validation_data=(testX, testY),
#               class_weight=classWeight, batch_size=BATCH_SIZE, epochs=EPOCHS,
#               callbacks=callbacks, verbose=2)

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                        validation_data=(testX, testY),
                        class_weight=classWeight,
                        steps_per_epoch=len(trainX) // BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks, verbose=2)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
# model.save(output_model_path)
model.save(output_model_path)
