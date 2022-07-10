#!/usr/bin/env python
# coding: utf-8

# In[3]:


import keras
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob


# In[4]:


folder = ["gu","choki","choki_m","pa","pa_m"]

image_size = 64

X = []
Y = []
for index, name in enumerate(folder):
    dir = "./data/" + name
    files = glob.glob(dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)


# In[5]:


X = X.astype('float32')
X = X / 255.0
#X = X.reshape((-1, 64, 64, 1))
#print(X.shape)


# In[6]:


# 正解ラベルの形式を変換
Y = np_utils.to_categorical(Y, 5)


# In[7]:


# 学習用データとテストデータ
X_train, Y_train = X, Y
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)


# In[8]:


print(X_train.shape, Y_train.shape)


# In[20]:


# CNNを構築
model = Sequential()
 
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))
 
# コンパイル
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])


# In[27]:


#訓練
#history = model.fit(X_train, Y_train, batch_size=32, epochs=8)
model.load_weights('jyanken1.h5')
history = model.fit(X_train, Y_train, batch_size=32, epochs=10)


# In[28]:


# モデルの保存
open('cnn_jyanken.json',"w").write(model.to_json())

# 学習済みの重みを保存
model.save_weights('jyanken1.h5')


# In[5]:


import cv2
import numpy as np
from keras.models import model_from_json

def save_frame_camera_cycle(device_num, delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)
    
    # モデルの読み込み
    model = model_from_json(open('cnn_jyanken.json', 'r').read())
    # 重みの読み込み
    #model.load_weights('appraisal.h5')
    model.load_weights('jyanken.h5')
    
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (390, 200), (610, 450), (255, 255, 0), thickness=2)
        
        
        img = frame[200:450, 390:610]
        img = cv2.resize(img, (64, 64))
        data = np.asarray(img)
        data = data.astype('float32')
        data = data / 255.0
        data = data.reshape((1, 64, 64, 3))
        ans_dat = model.predict(data)
        ans_dat = np.argmax(ans_dat)
        
        if ans_dat == 0:
            cv2.putText(frame, '{gu}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        if ans_dat == 1:
            cv2.putText(frame, '{choki}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        if ans_dat == 2:
            cv2.putText(frame, '{choki_m}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        if ans_dat == 3:
            cv2.putText(frame, '{pa}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        if ans_dat == 4:
            cv2.putText(frame, '{pa_m}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow(window_name, frame)
    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    save_frame_camera_cycle(1)


# In[ ]:




