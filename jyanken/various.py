#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from keras.models import model_from_json

def save_frame_camera_cycle(device_num, delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)
    
    # モデルの読み込み
    model = model_from_json(open('cnn_jyanken.json', 'r').read())
    # 重みの読み込み
    #model.load_weights('appraisal.h5')
    model.load_weights('jyanken05.h5')
    
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
            part = cv2.imread('jyanken_pa.png')
            part = cv2.resize(part, (100, 100))
            cv2.putText(frame, '{gu}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            frame[100:200, 100:200] = part
        if ans_dat == 1:
            part = cv2.imread('jyanken_gu.png')
            part = cv2.resize(part, (100, 100))
            cv2.putText(frame, '{choki}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            frame[100:200, 100:200] = part
        if ans_dat == 2:
            part = cv2.imread('jyanken_gu.png')
            part = cv2.resize(part, (100, 100))
            cv2.putText(frame, '{choki_m}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            frame[100:200, 100:200] = part
        if ans_dat == 3:
            part = cv2.imread('jyanken_choki.png')
            part = cv2.resize(part, (100, 100))
            cv2.putText(frame, '{pa}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            frame[100:200, 100:200] = part
        if ans_dat == 4:
            part = cv2.imread('jyanken_choki.png')
            part = cv2.resize(part, (100, 100))
            cv2.putText(frame, '{pa_m}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            frame[100:200, 100:200] = part

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow(window_name, frame)
    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    save_frame_camera_cycle(1)


# In[8]:


part = cv2.imread('jyanken_pa.png')
print(part)
alpha = part[:,:,3]
print(alpha)


# In[2]:


r = [1, 2, 4]
print(r[0])


# In[ ]:




