import os
import cv2
import numpy as np 
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from utils import DataGenerator
from utils import CTCLayer
print("Tensorflow version: ", tf.__version__)


seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)
data_dir = Path("C:\\Users\\Sai Krishna\\Downloads\\archive\\sastra_train")
images = list(data_dir.glob("*.png"))

characters = set()
captcha_length = []
dataset = []

sample_space="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcedfghijklmnopqrstuvwxyz"
for img_path in images:
    
    label = img_path.name.split(".png")[0]
    captcha_length.append(len(label))
    dataset.append((str(img_path), label))
    for ch in sample_space:
        characters.add(ch)
        

    
characters = sorted(characters)
dataset = pd.DataFrame(dataset, columns=["img_path", "label"], index=None)
dataset = dataset.sample(frac=1.).reset_index(drop=True)
dataset.head()
training_data, validation_data = train_test_split(dataset, test_size=0.1, random_state=seed)

training_data = training_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)

char_to_labels = {char:idx for idx, char in enumerate(characters)}

labels_to_char = {val:key for key, val in char_to_labels.items()}

# def is_valid_captcha(captcha):
#     for ch in captcha:
#         if not ch in characters:
#             return False
#     return True

def generate_arrays(df, resize=True, img_height=50, img_width=200):
    """Generates image array and labels array from a dataframe.
    
    Args:
        df: dataframe from which we want to read the data
        resize (bool)    : whether to resize images or not
        img_weidth (int): width of the resized images
        img_height (int): height of the resized images
        
    Returns:
        images (ndarray): grayscale images
        labels (ndarray): corresponding encoded labels
    """
    
    num_items = len(df)
    images = np.zeros((num_items, img_height, img_width), dtype=np.float32)
    labels = [0]*num_items
    
    for i in range(num_items):
        img = cv2.imread(df["img_path"][i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if resize: 
            img = cv2.resize(img, (img_width, img_height))
        
        img = (img/255.).astype(np.float32)
        label = df["label"][i]
        # if is_valid_captcha(label):
        images[i, :, :] = img
        labels[i] = label
    
    return images, np.array(labels)

training_data, training_labels = generate_arrays(df=training_data)
validation_data, validation_labels = generate_arrays(df=validation_data)

# class DataGenerator(keras.utils.Sequence):
#     """Generates batches from a given dataset.
    
#     Args:
#         data: training or validation data
#         labels: corresponding labels
#         char_map: dictionary mapping char to labels
#         batch_size: size of a single batch
#         img_width: width of the resized
#         img_height: height of the resized
#         downsample_factor: by what factor did the CNN downsample the images
#         max_length: maximum length of any captcha
#         shuffle: whether to shuffle data or not after each epoch
#     Returns:
#         batch_inputs: a dictionary containing batch inputs 
#         batch_labels: a batch of corresponding labels 
#     """
    
#     def __init__(self,
#                  data,
#                  labels,
#                  char_map,
#                  batch_size=16,
#                  img_width=200,
#                  img_height=50,
#                  downsample_factor=4,
#                  max_length=5,
#                  shuffle=True
#                 ):
#         self.data = data
#         self.labels = labels
#         self.char_map = char_map
#         self.batch_size = batch_size
#         self.img_width = img_width
#         self.img_height = img_height
#         self.downsample_factor = downsample_factor
#         self.max_length = max_length
#         self.shuffle = shuffle
#         self.indices = np.arange(len(data))    
#         self.on_epoch_end()
        
#     def __len__(self):
#         return int(np.ceil(len(self.data) / self.batch_size))
    
#     def __getitem__(self, idx):
#         curr_batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

#         batch_len = len(curr_batch_idx)

#         batch_images = np.ones((batch_len, self.img_width, self.img_height, 1),
#                                dtype=np.float32)
#         batch_labels = np.ones((batch_len, self.max_length), dtype=np.float32)
#         input_length = np.ones((batch_len, 1), dtype=np.int64) * \
#                                 (self.img_width // self.downsample_factor - 2)
#         label_length = np.zeros((batch_len, 1), dtype=np.int64)
        
        
#         for j, idx in enumerate(curr_batch_idx):

#             img = self.data[idx].T

#             img = np.expand_dims(img, axis=-1)

#             text = self.labels[idx]
   
#             if is_valid_captcha(text):
#                 label = [self.char_map[ch] for ch in text]
#                 batch_images[j] = img
#                 batch_labels[j] = label
#                 label_length[j] = len(text)
        
#         batch_inputs = {
#                 'input_data': batch_images,
#                 'input_label': batch_labels,
#                 'input_length': input_length,
#                 'label_length': label_length,
#                 }
#         return batch_inputs, np.zeros(batch_len).astype(np.float32)
        
    
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)


batch_size = 16


img_width=200
img_height=50 


downsample_factor=4


max_length=5


train_data_generator = DataGenerator(data=training_data,
                                     labels=training_labels,
                                     char_map=char_to_labels,
                                     batch_size=batch_size,
                                     img_width=img_width,
                                     img_height=img_height,
                                     downsample_factor=downsample_factor,
                                     max_length=max_length,
                                     shuffle=True
                                    )


valid_data_generator = DataGenerator(data=validation_data,
                                     labels=validation_labels,
                                     char_map=char_to_labels,
                                     batch_size=batch_size,
                                     img_width=img_width,
                                     img_height=img_height,
                                     downsample_factor=downsample_factor,
                                     max_length=max_length,
                                     shuffle=False
                                    ) 

# class CTCLayer(layers.Layer):
#     def __init__(self, name=None):
#         super().__init__(name=name)
#         self.loss_fn = keras.backend.ctc_batch_cost

#     def call(self, y_true, y_pred, input_length, label_length):
#         # Compute the training-time loss value and add it
#         # to the layer using `self.add_loss()`.
#         loss = self.loss_fn(y_true, y_pred, input_length, label_length)
#         self.add_loss(loss)
        
#         # On test time, just return the computed loss
#         return loss



def build_model():
    input_img = layers.Input(shape=(img_width, img_height, 1),
                            name='input_data',
                            dtype='float32')
    labels = layers.Input(name='input_label', shape=[max_length], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    
    # First conv block
    x = layers.Conv2D(32,
               (3,3),
               activation='relu',
               kernel_initializer='he_normal',
               padding='same',
               name='Conv1')(input_img)
    x = layers.MaxPooling2D((2,2), name='pool1')(x)
    
    # Second conv block
    x = layers.Conv2D(64,
               (3,3),
               activation='relu',
               kernel_initializer='he_normal',
               padding='same',
               name='Conv2')(x)
    x = layers.MaxPooling2D((2,2), name='pool2')(x)
    

    new_shape = ((img_width // 4), (img_height // 4)*64)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2)(x)
    
    # RNNs
    x = layers.Bidirectional(layers.LSTM(128,
                                         return_sequences=True,
                                         dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64,
                                         return_sequences=True,
                                         dropout=0.25))(x)
    
    # Predictions
    x = layers.Dense(len(characters)+1,
              activation='softmax', 
              name='dense2',
              kernel_initializer='he_normal')(x)
    
    # Calculate CTC
    output = CTCLayer(name='ctc_loss')(labels, x, input_length, label_length)
    
    # Define the model
    model = keras.models.Model(inputs=[input_img,
                                       labels,
                                       input_length,
                                       label_length],
                                outputs=output,
                                name='ocr_model_v1')
    
    # Optimizer
    sgd = keras.optimizers.SGD(learning_rate=0.002,
                    
                               momentum=0.9,
                               nesterov=True,
                               clipnorm=5)
    
    # Compile the model and return 
    model.compile(optimizer=sgd)
    return model

model = build_model()
# Add early stopping to stop overfitting
# es = keras.callbacks.EarlyStopping(monitor='val_loss',
#                                    patience=5,
#                                    restore_best_weights=True)

# Train the model
model.fit(train_data_generator,
                    validation_data=valid_data_generator,
                    epochs=50)

model.save("C:\\Users\\Sai Krishna\\Desktop\\captcha\\sajith.h5")