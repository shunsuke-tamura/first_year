#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import random
import numpy as np
from time import sleep
from pygame import mixer
from keras.models import model_from_json


i = 0
r = []
while True:
    n = random.randint(1, 10)
    if not n in r:
        i += 1
        r.append(n)
        if i == 3:
            break
        
        
def play(f, f2, f3):
    if f == 1:
        mixer.init()        #初期化
        mixer.music.load("jyanken1.mp3")
        mixer.music.play(1)
        
    if f == 2:
        mixer.init()        #初期化
        mixer.music.load("jyanken2.mp3")
        mixer.music.play(1)
        
    if f == 3:
        mixer.init()        #初期化
        mixer.music.load("opening2.mp3")
        mixer.music.play(1)
        
    if f == 4:
        rand = r[f2]
        mixer.init()        #初期化
        mixer.music.load("question/" + "{}".format(rand) + ".mp3")
        mixer.music.play(1)
        
    if f == 5:
        rand = random.randint(1, 5)
        mixer.init()        #初期化
        mixer.music.load("reaction/" + "{}".format(rand) + ".mp3")
        mixer.music.play(1)
        
    if f == 6:
        mixer.init()        #初期化
        mixer.music.load("start/" + "{}".format(f2) + ".mp3")
        mixer.music.play(1)
        
    if f == 7:
        if f3 == 0:
            rand = random.randint(1, 3)
            mixer.init()        #初期化
            mixer.music.load("result/" + "{}".format(rand) + ".mp3")
            mixer.music.play(1)
        elif f3 == 1:
            rand = random.randint(4, 6)
            rand = 4
            mixer.init()        #初期化
            mixer.music.load("result/" + "{}".format(rand) + ".mp3")
            mixer.music.play(1)
        elif f3 == 2:
            rand = 7
            mixer.init()        #初期化
            mixer.music.load("result/" + "{}".format(rand) + ".mp3")
            mixer.music.play(1)
        
    if f == 8:
        mixer.init()        #初期化
        mixer.music.load("ending.mp3")
        mixer.music.play(1)
        
    if f == 9:
        mixer.init()        #初期化
        mixer.music.load("q_start/" + "{}".format(f2) + ".mp3")
        mixer.music.play(1)
        

def display(frame, ans_dat):
    if ans_dat == 0:
        part = cv2.imread('jyanken_pa.png')
        part = cv2.resize(part, (100, 100))
        frame[50:150, 450:550] = part
    if ans_dat == 1:
        part = cv2.imread('jyanken_gu.png')
        part = cv2.resize(part, (100, 100))
        frame[50:150, 450:550] = part
    if ans_dat == 2:
        part = cv2.imread('jyanken_gu.png')
        part = cv2.resize(part, (100, 100))
        frame[50:150, 450:550] = part
    if ans_dat == 3:
        part = cv2.imread('jyanken_choki.png')
        part = cv2.resize(part, (100, 100))
        frame[50:150, 450:550] = part
    if ans_dat == 4:
        part = cv2.imread('jyanken_choki.png')
        part = cv2.resize(part, (100, 100))
        frame[50:150, 450:550] = part
        
    return frame

def judg(data, model):
    data = data.astype('float32')
    data = data / 255.0
    data = data.reshape((1, 64, 64, 3))
    ans_dat = model.predict(data)
    ans_dat = np.argmax(ans_dat)
    
    return ans_dat
    
    
def main(device_num):
    cap = cv2.VideoCapture(device_num)
    
    if not cap.isOpened():
        return
    
    n = 0
    f1 = 0
    f2 = 0
    f3 = 0
    f4 = 0
    # モデルの読み込み
    model = model_from_json(open('cnn_jyanken.json', 'r').read())
    # 重みの読み込み
    model.load_weights('jyanken.h5')
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (390, 200), (610, 450), (255, 255, 0), thickness=2)
        
        if f1 == 0:
            img = frame[200:450, 390:610]
            img = cv2.resize(img, (64, 64))
            data = np.asarray(img)
            ans_dat = judg(data, model)
            if ans_dat == 3:
                f1 = 1
                play(3, f2, f3)
                
        if f1 == 1:
            n += 1
        if f1 == 1 and n == 730:
            play(4, f2, f3)
            ans_dat == 10
            n = 0
            f1 = 2
        if f1 == 2:
            img = frame[200:450, 390:610]
            img = cv2.resize(img, (64, 64))
            data = np.asarray(img)
            ans_dat = judg(data, model)
            n += 1
            if ans_dat == 1 and n > 70:
                play(5, f2, f3)
                n = 0
                f1 = 3
            if ans_dat == 3 and n > 70:
                play(5, f2, f3)
                n = 0
                f1 = 3
        if f1 == 3:
            n += 1
        if f1 == 3 and n == 150:
            play(6, f2, f3)
            n = 0
            f1 = 4
        if f1 == 4:
            n += 1
        if f1 == 4 and n == 330:
            if f2 < 2:
                play(1, f2, f3)
            if f2 == 2:
                play(2, f2, f3)            
            n = 0
            f1 = 5
        if f1 == 5:
            n += 1
        if f1 == 5 and f2 < 2 and n == 135:
            r = [0, 1, 3]
            ans_dat = r[random.randint(0, 2)]
            n = 0
            f1 = 6
        if f1 == 5 and f2 == 2 and n == 140:
            img = frame[200:450, 390:610]
            img = cv2.resize(img, (64, 64))
            data = np.asarray(img)
            ans_dat = judg(data, model)
            #print(ans_dat)
            n = 0
            f1 = 6
        if f1 == 6:
            n += 1
            frame = display(frame, ans_dat)
            if n == 10:
                img = frame[200:450, 390:610]
                cv2.imwrite("img" + "{}".format(f2) + ".jpg", img)
                img = cv2.resize(img, (64, 64))
                data = np.asarray(img)
                ans_dat1 = judg(data, model)
            if n == 90:
                #print(ans_dat)
                if ans_dat1 == ans_dat and f4 == 1:
                    f3 == 2
                if ans_dat1 == ans_dat and f4 == 0:
                    f3 = 1
                    f4 = 1
                n = 0
                f1 = 7
        if f1 == 7:
            if n == 0 and f2 < 2:
                play(7, f2, f3)
                f3 = 0
            elif n == 0 and f2 == 2:
                n = 149
            n += 1
        if f1 == 7 and n == 150:
            if f2 < 2:
                play(9, f2, f3)
            n = 570
            f1 = 1
            f2 += 1
        if f2 == 3:
            f1 = 8
            if n == 570:
                play(8, f2, f3)
                n = 0
            n += 1
        if f2 == 3 and n == 330:
            break
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('JYAIKO', frame)
    cv2.destroyWindow('JYAIKO')
    
    
    
if __name__ == '__main__':
    main(0)


# In[ ]:





# In[ ]:




