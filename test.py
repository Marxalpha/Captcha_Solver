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
from utils import DataGenerator
from utils import CTCLayer
from utils import generate_arrays
from utils import *

batch_size = 16
img_width=200
img_height=50 
downsample_factor=4
max_length=5


test_data_dir=Path("C:\\Users\\Sai Krishna\\Downloads\\archive\\test")
test_images=list(test_data_dir.glob("*.png"))
test_dataset=[]
for img_path in test_images:
    label = img_path.name.split(".png")[0]
    # captcha_length.append(len(label))
    test_dataset.append((str(img_path),label))
test_dataset = pd.DataFrame(test_dataset, columns=["img_path", "label"], index=None)
test_data = test_dataset.reset_index(drop=True)
test_data,test_labels=generate_arrays(df=test_data)

test_data_generator = DataGenerator(data=test_data,
                                     labels=test_labels,
                                     char_map=char_to_labels,
                                     batch_size=batch_size,
                                     img_width=img_width,
                                     img_height=img_height,
                                     downsample_factor=downsample_factor,
                                     max_length=max_length,
                                     shuffle=False
                                   )
prediction_model = keras.models.load_model("sajith.h5")


def decode_batch_predictions(pred):
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0])*pred.shape[1]
    
    
    results = keras.backend.ctc_decode(pred, 
                                        input_length=input_len,
                                        greedy=True)[0][0]
    
   
    output_text = []
    for res in results.numpy():
        outstr = ''
        for c in res:
            if c < len(characters) and c >=0:
                outstr += labels_to_char[c]
        output_text.append(outstr)
    return output_text


for p, (inp_value, _) in enumerate(test_data_generator):
    bs = inp_value['input_data'].shape[0]
    X_data = inp_value['input_data']
    labels = inp_value['input_label']
    preds = prediction_model.predict(X_data)
    pred_texts = decode_batch_predictions(preds)
    orig_texts = []
    for label in labels:
        text = ''.join([labels_to_char[int(x)] for x in label])
        orig_texts.append(text)
        
    for i in range(bs):
        print(f'Ground truth: {orig_texts[i]} \t Predicted: {pred_texts[i]}')
    break