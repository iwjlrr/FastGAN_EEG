import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#file_path ='D:/AI/BCI/project/EEG_ETR/pytorch-cnn-cifar10-master/data/my_train/1/5.png'
#ima = Image.open(file_path)
#print(ima.size)
# 超参数设置
num_epochs = 10
num_classes = 36
batch_size = 64
learning_rate = 0.0001
import torchvision.datasets as dset

classes = ('1','2','3','4','5','6','7','8','9')


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#  训练数据集
train_dataset = dset.ImageFolder(root='./Mental Task/Mental Arithmetic task/my_train',
                                             transform=transform)


# 测试数据集
test_dataset = dset.ImageFolder(root='./data/my_test',
                                            transform=transform)

# 训练数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
# 查看数据,取一组batch
#data_iter = iter(test_loader)

#images, labels = next(data_iter)
# 取batch中的一张图像
#idx = 15
#image = images[idx]
#image = Image.open(image)
#print(image.size)
#image = np.transpose(image, (1,2,0))
#plt.imshow(image)
#classes[labels[idx].numpy()]


# 搭建卷积神经网络模型

class ConvNet(nn.Module):
    def __init__(self, num_classes=9):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            # 卷积层计算
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            #  批归一化
            nn.BatchNorm2d(32),
            # ReLU激活函数
            nn.ReLU(),
            # 池化层：最大池化
            nn.MaxPool2d(kernel_size=2, stride=2))
#100
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
#50

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(8 * 8 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)



# 定义前向传播顺序
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
       # out = self.conv4(out)
        #out = self.conv5(out)
        #print('reshape 前的size:')
        #print(out.size())
        #print('reshape 前的size(0):')
        #print(out.size(0))
        out = out.reshape(out.size(0), -1)
        #print('rehsape 后的size:')
        #print(out.size())
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# 实例化一个模型，并迁移至gpu
model = ConvNet(num_classes).to(device)
# 定义损失函数和优化器

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)

print(total_step)

# save or show losses
losses = []  # record losses
def save_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.savefig('CNN_loss.png')
    # plt.show()


for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # 注意模型在GPU中，数据也要搬到GPU中
        images = images.to(device)
        #print(images.size())
        labels = labels.to(device)
        #print(labels.size())

        #print(images)
        #print(images)

        # 前向传播
        outputs = model(images)
        #print('outputs')
        #print(outputs.size())
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            loss = running_loss / 100
            losses.append(loss)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss))
            running_loss = 0.0
        # save losses
    save_losses(losses)



# 节省计算资源，不去计算梯度
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the train images: {:.2f} %'.format(100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(9):
            label = labels[i]
            class_correct[label] += c[i].item()

            class_total[label] += 1

for i in range(9):
    print('train every class Accuracy of %5s : %.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

#  保存模型
#torch.save(model.state_dict(), 'model.ckpt')


