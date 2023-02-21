
from My_ResNet import model

from torch.autograd import Variable
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
batch_size = 32
# 设备设置

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cifar10 分类索引
#classes = ('plane', 'car', 'bird', 'cat',
 #          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#  训练数据集
#train_dataset = dset.ImageFolder(root='./data/train',
#                                             transform=transform)


# 测试数据集
test_dataset = dset.ImageFolder(root='./data_32/my_test_GAN',

                                            transform=transform)

#print(train_dataset.classes)  #根据分的文件夹的名字来确定的类别
#print(train_dataset.class_to_idx) #按顺序为这些类别定义索引为0,1...
#print(train_dataset.imgs) #返回从所有文件夹中得到的图片的路径以及其类别




# 训练数据加载器
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
 #                                          batch_size=batch_size,
 #                                          shuffle=True)
# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
net = model.to(device)
net.load_state_dict(torch.load('ResNet_MI_32_model.ckpt'))
net.eval()


correct = []
def save_accuracy(correct):
    n = np.arange(correct.shape())
    plt.plot(n, correct)
    plt.savefig('My_ResNet_MI_32_test_accuracy.png')

# 节省计算资源，不去计算梯度
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        images = Variable(images)
        labels = labels.to(device)
        labels = Variable(labels)
        #print(labels)

        outputs = net(images)
        predicted = torch.max(outputs.data, 1)[1].cpu().numpy()
        total += labels.size(0)
        correct += ((predicted == labels.cpu().numpy()).astype(int).sum())/total
        correct+= correct.item()
        correct = correct / 10
        correct.append(correct)
    save_accuracy(correct)
    print('Test Accuracy of the model on the test images : {:.2f} %'.format(correct))
'''

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        print(labels)
       
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()

            class_total[label] += 1

for i in range(10):
    print('test every class Accuracy of %5s : %.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
'''




