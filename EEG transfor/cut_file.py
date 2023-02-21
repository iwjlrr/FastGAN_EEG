import os
import random
import shutil


# source_file:源路径, target_ir:目标路径
def cover_files(source_dir, target_ir):
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)

        if os.path.isfile(source_file):
            shutil.copy(source_file, target_ir)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def moveFile(file_dir, save_dir):
    ensure_dir_exists(save_dir)
    path_dir = os.listdir(file_dir)  # 取图片的原始路径
    filenumber = len(path_dir)
    rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(path_dir, picknumber)  # 随机选取picknumber数量的样本图片
    # print (sample)
    for name in sample:
        shutil.move(file_dir + name, save_dir + name)


if __name__ == '__main__':
    file_dir = 'D:\AI\BCI\project\EEG_ETR\pytorch-cnn-cifar10-master\Mental Task\Rest\my_train\sub27/'  # 源图片文件夹路径
    save_dir = 'D:\AI\BCI\project\EEG_ETR\pytorch-cnn-cifar10-master\Mental Task\Rest\my_test\sub27/'  # 移动到新的文件夹路径
    moveFile(file_dir,save_dir)


'''
import os, sys
import random
import shutil


def copyFile(fileDir):
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir, 300)
    print(sample)
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)


if __name__ == '__main__':
    # open /textiles
    path = "/home/fairy/workspace/dataset/textiles/"
    dirs = os.listdir(path)
    i = 0
    # output all folds
    for file in dirs:
        print(file)
        i = i + 1
        filename = "/home/fairy/workspace/dataset/Fabric" + str(i)
        os.mkdir(filename)
        fileDir = path + "Fabric" + str(i) + "/"
        tarDir = filename + "/"
        copyFile(fileDir)
'''