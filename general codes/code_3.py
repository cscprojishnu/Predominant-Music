import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import cv2
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import *
import keras
from tensorflow.keras.callbacks import Callback

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score,RocCurveDisplay

SEED = 8
IMG_SIZE = 256, 256
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE

def create_images_list(path):
    full_path = []
    images = sorted(os.listdir(path))
    for i in images:
        full_path.append(os.path.join(path, i))
    return full_path


classes = {0: 'Benign', 1: 'Malignant'}

train_bening_imgs = create_images_list(r"D:\DJT\Downloads\PAPER\PAPER 3\testset")
train_malignant_imgs = create_images_list(r"D:\DJT\Downloads\PAPER\PAPER 3\trainingset")

full_data = pd.concat([pd.DataFrame({'image' : train_bening_imgs, 'label': 0 }),
                      pd.DataFrame({'image' : train_malignant_imgs, 'label': 1 })])
# shuffling dataset
full_data = full_data.sample(frac = 1, ignore_index = True, random_state = SEED)

# train and valid splitting
train_data, valid_data = train_test_split(full_data, test_size = 0.2, stratify = full_data['label'])

train_data = train_data.reset_index(drop = True)
valid_data = valid_data.reset_index(drop = True)

# test dataframe
test_bening_imgs = create_images_list('/kaggle/input/melanoma-cancer-dataset/test/Benign')
test_malignant_imgs = create_images_list('/kaggle/input/melanoma-cancer-dataset/test/Malignant')

test_data = pd.concat([pd.DataFrame({'image' : test_bening_imgs, 'label': 0 }),
                       pd.DataFrame({'image' : test_malignant_imgs, 'label': 1 })])
test_data = test_data.sample(frac = 1, ignore_index = True, random_state = SEED)


print('total train images \t{0}'.format(train_data.shape[0]))
print('total valid images \t{0}'.format(valid_data.shape[0]))
print('total test images \t{0}'.format(test_data.shape[0]))

def img_preprocessing(image, label):
    img = tf.io.read_file(image)
    img = tf.io.decode_png(img, channels = 3)
    img = tf.image.resize(img, size =(IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    
    
    return img, label


# Data augmentation 
img_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation((-0.3, 0.3), interpolation="bilinear"),
    RandomContrast(0.03),
    RandomCrop(*IMG_SIZE)])

train_loader = tf.data.Dataset.from_tensor_slices((train_data['image'], train_data['label']))
train_dataset = (train_loader
                 .map(img_preprocessing, num_parallel_calls = AUTO)
                 .shuffle(BATCH_SIZE*10)
                 .batch(BATCH_SIZE)
                 .map(lambda img, label: (img_augmentation(img), label), num_parallel_calls =AUTO)
                 .prefetch(AUTO))


valid_loader = tf.data.Dataset.from_tensor_slices((valid_data['image'], valid_data['label']))
valid_dataset = (valid_loader
                 .map(img_preprocessing, num_parallel_calls = AUTO)
                 .batch(BATCH_SIZE)
                 .prefetch(AUTO))

test_loader = tf.data.Dataset.from_tensor_slices((test_data['image'], test_data['label']))
test_dataset = (test_loader
                 .map(img_preprocessing, num_parallel_calls = AUTO)
                 .batch(BATCH_SIZE)
                 .prefetch(AUTO))

class Involution(Layer):
    def __init__(self, channel, group_num, kernel_size, stride, reduce_ratio, **kwargs):
        super(Involution, self).__init__(**kwargs)
        self.channel = channel
        self.group_num = group_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduce_ratio = reduce_ratio


    def build(self, input_shape):
        (_, height, width, num_channels) = input_shape
        height = height // self.stride
        width = width // self.stride

        self.stride_layer = AveragePooling2D(pool_size = self.stride, strides = self.stride, padding = 'same')
        self.kernel_gen = keras.Sequential([ 
            Conv2D(filters = self.channel // self.reduce_ratio , kernel_size = 1),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters = self.kernel_size*self.kernel_size*self.group_num, kernel_size = 1)   ])

        self.kernel_reshape = Reshape(target_shape=(height, width, self.kernel_size*self.kernel_size, 1, self.group_num))
        self.input_patches_reshape = Reshape(target_shape=(height, width, self.kernel_size*self.kernel_size, 
                                                           num_channels // self.group_num, self.group_num))

        self.output_reshape = Reshape(target_shape=(height, width, num_channels))



    def call(self, inputs):
        kernel_input = self.stride_layer(inputs)
        kernel = self.kernel_gen(kernel_input)

        kernel = self.kernel_reshape(kernel)

        input_patches = tf.image.extract_patches(images = inputs,
                                                 sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                 strides=[1, self.stride, self.stride, 1],
                                                 rates=[1, 1, 1, 1], padding="SAME")
    
        input_patches = self.input_patches_reshape(input_patches)

        output = tf.multiply(kernel, input_patches)
        output = tf.reduce_sum(output, axis=3)

        output = self.output_reshape(output)

        return output
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "channel" : self.channel,
                "group_num": self.group_num,
                "kernel_size" : self.kernel_size,
                "stride" : self.stride,
                "reduce_ratio" : self.reduce_ratio}
    

# a custom callback for evaluation of test set during training each 10 loop
class Test_Pred_During_Training(Callback):
    
    def __init__(self, model):
        self.model = model
        
    def on_epoch_end(self, epochs, logs = None ):
        try:
            if (epochs+1) % 10 == 0:
                test_pred = self.model.predict(test_dataset, verbose = 0)
                test_pred = np.argmax(test_pred, axis = 1)

                mse = mean_squared_error(test_data['label'], test_pred)
                f1 = f1_score(test_data['label'], test_pred, average = 'weighted')
                acc = accuracy_score(test_data['label'], test_pred)

                print('\nMean Squared Error : {0:.5f}'.format(mse))
                print('Weighted F1 Score : {0:.3f}'.format(f1))
                print('Accuracy Score : {0:.3f} %'.format(acc*100))

                print("--"*40)
        except ValueError:
            pass

inp = Input(shape=(*IMG_SIZE, 3))
    
X = Involution(channel = 32, group_num=3, kernel_size=3, stride=1, reduce_ratio=3, name="Involution_1")(inp)
X = Activation('elu')(X)
X = MaxPooling2D(2)(X)
    
X = Involution(channel = 32, group_num=3, kernel_size=3, stride=1, reduce_ratio=3, name="Involution_2")(X)
X = Activation('elu')(X)
X = MaxPooling2D(2)(X)

X = Involution(channel = 32, group_num=3, kernel_size=3, stride=1, reduce_ratio=3, name="Involution_3")(X)
X = Activation('elu')(X)
X = MaxPooling2D(2)(X)

X = Flatten()(X)
X = Dense(128, activation="relu")(X)
out = Dense(2)(X)

model = keras.Model(inputs=inp, outputs=out)

model.compile(optimizer = tf.optimizers.SGD(learning_rate = 0.0007, weight_decay = 0.05, momentum = 0.9),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["acc"])

my_callbacks = [Test_Pred_During_Training(model),
                tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',factor=0.1, min_delta = 0.01, patience=6),
                tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.01,patience=20),
                tf.keras.callbacks.ModelCheckpoint("/kaggle/working/my_model.tf", monitor="val_loss", mode="min", save_best_only=True, verbose=1)]

model.summary()

hist = model.fit(train_dataset, epochs = 85, validation_data = valid_dataset, callbacks = [my_callbacks])

fig, axs = plt.subplots(1,2, figsize = (10, 4), dpi = 100)

axs[0].grid(linestyle = 'dashdot')
axs[0].plot(hist.history['loss'])
axs[0].plot(hist.history['val_loss'])
axs[0].set_xlabel('epochs', fontsize = 10)
axs[0].legend(['train loss', 'validation loss'], fontsize = 10)


axs[1].grid(linestyle = 'dashdot')
axs[1].plot(hist.history['acc'])
axs[1].plot(hist.history['val_acc'])
axs[1].set_xlabel('epochs', fontsize = 10)
axs[1].legend(['train accuracy', 'validation accuracy'], fontsize = 10)

# Predictions and scores
test_pred = model.predict(test_dataset)
test_pred = np.argmax(test_pred, axis = 1)

mse = mean_squared_error(test_data['label'], test_pred)
f1 = f1_score(test_data['label'], test_pred, average = 'weighted')
acc = accuracy_score(test_data['label'], test_pred)

print('Mean Squared Error : {0:.5f}'.format(mse))
print('Weighted F1 Score : {0:.3f}'.format(f1))
print('Accuracy Score : {0:.3f} %'.format(acc*100))

clf = classification_report(test_data['label'], test_pred, target_names = list(classes.values()))
print(clf)


cm = confusion_matrix(test_data['label'], test_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels = list(classes.values()))

fig, ax = plt.subplots(figsize=(5,5))
cmd.plot(ax=ax,  cmap = 'BuPu', colorbar = False)

test_take1 =  test_dataset.take(-1)
test_take1_ = list(test_take1)

# A function that creating 5 random images in the test set and predictions

# Red title -> a false prediction
# Green title -> a true prediction

def random_test_sample_with_prediction(SEED):
    idxs = np.random.default_rng(seed=SEED).permutation(len(test_pred))[:5]
    batch_idx = idxs // BATCH_SIZE
    image_idx = idxs-batch_idx * BATCH_SIZE
    idx = idxs

    fig, axs = plt.subplots(1,5, figsize = (12,12) ,dpi = 150)

    for i in range(5):
        img = test_take1_[batch_idx[i]][0][image_idx[i]]
        label = test_take1_[batch_idx[i]][1][image_idx[i]].numpy()
        
        if int(test_pred[idx[i]]) == label:
            axs[i].imshow(img, cmap = 'gray') 
            axs[i].axis('off')
            axs[i].set_title('image (no: ' + str(idx[i])  + ')' + '\n TRUE: ' + classes[label]
                             + '\n PRED: ' + classes[test_pred[idx[i]]]
                             , fontsize = 8, color = 'green')
        else:
            axs[i].imshow(img,  cmap = 'gray')
            axs[i].axis('off')
            axs[i].set_title('image (no: ' + str(idx[i])  + ')' + '\n TRUE: ' + classes[label]
                             + '\n PRED: ' + classes[test_pred[idx[i]]]
                             , fontsize = 8, color = 'red')
            
# Red title -> a false prediction
# Green title -> a true prediction

random_test_sample_with_prediction(10)
random_test_sample_with_prediction(104)
random_test_sample_with_prediction(193)


