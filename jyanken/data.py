#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import concurrent.futures
from pygame import mixer
from time import sleep



def play():
    mixer.init()        #初期化
    mixer.music.load("jyanken1.mp3")
    sleep(1)
    mixer.music.play(1)
    
    
def camera(device_num, dir_path, basename, cycle, a, delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    
    n = 0
    m = 0
    i = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (390, 200), (610, 450), (255, 255, 0), thickness=2)
        cv2.imshow(window_name, frame)
        if m == 0:
            play()
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        if m >=127 and m <= 137:
            if n == cycle:
                i += 1
                n = 0
                img = frame[200:450, 390:610]
                cv2.imwrite('{}'.format(dir_path) +'{}_{}.jpg'.format(a, i), img)
            n += 1
        m += 1
        if m == 160:
            m = 0
            n = 0
            i = 0
            a += 1
            if a == 121:
                break
    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    save_frame_camera_cycle(1, 'data/gu/', 'camera_capture_cycle', 1, 71)
    


# In[7]:


from pygame import mixer
from time import sleep
mixer.init()        #初期化
mixer.music.load("jyanken1.mp3")
mixer.music.play(1)
sleep(4)
print("ポン")


# In[5]:


import cv2
import os
import concurrent.futures
from pygame import mixer
from time import sleep



def play():
    mixer.init()        #初期化
    mixer.music.load("jyanken1.mp3")
    sleep(1)
    mixer.music.play(1)
    
    
def camera(device_num, dir_path, basename, cycle, a, delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    
    n = 0
    m = 0
    i = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (390, 200), (610, 450), (255, 255, 0), thickness=2)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        if n == cycle:
            n = 0
            i += 1
            img = frame[200:450, 390:610]
            cv2.imwrite('{}'.format(dir_path) +'{}_{}_{}.jpg'.format(i, i, i), img)
        n += 1
        m += 1
        if m == 1201:
            break
    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    camera(1, 'data/guu/', 'camera_capture_cycle', 10, 1)
    


# In[ ]:




