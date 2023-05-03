import os
import cv2

path = "/home/bo/桌面/LSVT/train_origin/train_full_images_0/"
datanames = os.listdir(path)
list = []
for i in datanames:
    list.append(i)
# print(list)

i=0
scale = 2.4
for img_name in list:
    img = cv2.imread(path+img_name)
    h,w,c = img.shape

    if h/w>1.2 or w/h>1.2 or h/512>scale or 512/h>scale or w/512>scale or 512/w>scale or min(h,w)/512<1:
        continue
    img_test = cv2.resize(img, (512, 512))

    cv2.imwrite('/home/bo/桌面/LSVT/train_32/'+img_name, img_test)

    print(str(i)+'/'+str(len(list))+': '+img_name)
    i+=1