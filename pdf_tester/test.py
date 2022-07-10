import cv2
import glob
import numpy as np

t_imgs = []
f_imgs = []
for name in ["mine", "true"]:
    dir = "./img/" + name
    files = glob.glob(dir + "/*.jpg")
    for file in files:
        image = cv2.imread(file)
        data = np.asarray(image)
        if name == "true":
            t_imgs.append(data)
        else:
            f_imgs.append(data)

diff_list = []
for t_img, f_img in zip(t_imgs, f_imgs):
    diff = t_img - f_img
    diff_list.append(diff)

for i, res in enumerate(diff_list):
    cv2.imwrite("./result/result_{}.jpg".format(i), res)
