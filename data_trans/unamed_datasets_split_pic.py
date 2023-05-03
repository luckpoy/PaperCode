#coding=utf-8
import os
import shutil
import traceback

from numpy import arange
 
 
def move_file(src_path, dst_path, file):
    print(file)
    print ('from : ',src_path)
    print ('to : ',dst_path)
    try:
        # cmd = 'chmod -R +x ' + src_path
        # os.popen(cmd)
        f_src = os.path.join(src_path, file)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        f_dst = os.path.join(dst_path, file)
        shutil.move(f_src, f_dst)
    except Exception as e:
        print ('move_file ERROR: ',e)
        traceback.print_exc()

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
		print ("---  OK  ---")
	else:
		print ("---  There is this folder!  ---")
		

path = "/home/bo/桌面/LSVT/train_32/"
datanames = os.listdir(path)
list = []
for i in datanames:
    list.append(i)
# print(list)

dest_path = "/home/bo/桌面/LSVT/train/"

index = 0
i = 0
new_file = ""
for img in list:

    if index % 100 == 0:

        new_file = dest_path + str(i)
        mkdir(new_file)
        i=i+1

    move_file(path,new_file,img)

    index = index + 1
