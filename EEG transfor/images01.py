import os
from PIL import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

base_dir = "D:\AI\BCI\dataset\eeg-during-mental-arithmetic-tasks-1.0.0\\rest\\train\sub27/"
filename = os.listdir(base_dir)
new_dir = "D:\AI\BCI\project\EEG_ETR\pytorch-cnn-cifar10-master\Mental Task\Rest\my_train/"

file_nums = sum([len(files) for root,dirs,files in os.walk(base_dir)])
print(file_nums)



i = 0
for img,i in zip(filename,range(10)):

    picture = io.imread(base_dir + img)
    #picture = Image.open(base_dir + img)
    slice_width, slice_height, _ = picture.shape

    width_crop = (slice_width - 410) // 2
    height_crop = (slice_height - 410) // 2
    if width_crop > 0:
        img_data = picture[width_crop:-width_crop, :, :]
    if height_crop > 0:
        img_data = img_data[:, height_crop:-height_crop,:]
    io.imsave(new_dir+'/'+np.str(i)+'.png',img_data)
    if i % 100 == 0:
        print('{} of {} have been cut'.format(i,file_nums))
    i = i + 1


i = 0
size_m = 200
size_n = 200
filename1 =  os.listdir(new_dir)
for img in filename1:
    image = Image.open(new_dir + img)
    image_size = image.resize((size_m, size_n), Image.ANTIALIAS)
    image_size.save(new_dir + img)
    if i % 100 == 0:
        print('{} of {} have been done'.format(i,file_nums))
    i = i + 1



'''
#picture = io.imread("D:/AI/BCI/project/EEG_ETR/pytorch-cnn-cifar10-master/data/my_test/4/5.png")# 图片路径
#file_path ='D:/AI/BCI/dataset/BCI Competition IV/BCICIV_2a_gdf/A02T images/5.png'
#ima = Image.open(file_path)
#plt.show(ima)
#print(picture.shape)
#io.imshow(picture)

#plt.show()
'''