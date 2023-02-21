import os

files = os.listdir("D:\pytorch-fid-master\path\To\Fake\sub7")#原来文件夹的路径
i = 0

for file in files:
    original = "D:\pytorch-fid-master\path\To\Fake\sub7" + os.sep + files[i]
    #修改后放置图片的路径 F:/ns，也可将 img_ 换成其他标注
    new = "D:\pytorch-fid-master\path\To\Fake\sub7" + os.sep + "H" + str(i + 1) + ".jpg"
    os.rename(original, new)
    i += 1





