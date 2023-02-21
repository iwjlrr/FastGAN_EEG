import mne
import numpy as np
import scipy.io
#%pylab inline
import matplotlib.pyplot as plt
data = scipy.io.loadmat('E:\Results-0628\BCI\BCI Dataset\BCI  Competition IV 1\BCICIV_calib_ds1a.mat') #读取mat文件
# print(data['cnt']) #数据集包括 x_train x_test y_train
#
#data = data['cnt']
#print(data)
#data = data['samples']
print(data)
#data = np.array(data)
#print(data.shape)
#data = data.T
#data_1=data['cnt'].T
#a=np.array([data_1[0,:]])
#b=np.array([data_1[1,:]])
#c=np.array([data_1[2,:]])

#print(a,a.shape)
#d=np.squeeze(np.array([a,b,c]))
#print(d,d.shape)
#data = np.delete(data,[0,17,18],axis=0)
# data = np.delete(data,[16,17,18,19,20,21,22,23,33,34,35,36,37,38,39,40],axis=0)
# print(data.shape)
# col_rand_data = np.arange(data.shape[1])
# print(col_rand_data)
# np.random.shuffle(col_rand_data)
# data = data[:,col_rand_data[0:10000]]
# print(data.shape)


# samples = np.samples.reshape(140,3,1152)

#print(samples) #140个被试，每个被试采集了C3、C4、CZ三个通道的脑电信号，每个通道采集了1152个点，采集了9s
# print(samples[:][0][0])#第一个被试者的C3通道脑电数据
#
# ch_names = ['AF3','AF4','F5','F3','F1','Fz','F2','F4','F6',
#             'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
#             'T7','C5','C3','C1','Cz',
#             'C2','C4','C6','T8','CP5','CP3','CP1','CPz','CP2','CP4','CP6','P5',
#             'P3','P1','Pz','P2','P4','P6','PO1','PO2','O1','O2'] #通道名称
# #ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
#  #           'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
# #            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
# #            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
# #            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
#  #           'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg', ]
# sfreq = 1000 #采样率
# info = mne.create_info(ch_names=ch_names, sfreq = sfreq,ch_types = 'eeg') #创建信号的信息
# info.set_montage('standard_1020')
# evoked = mne.EvokedArray(data, info)
# evoked.plot_sensors(ch_type='eeg')
# plt.show()
# for i in range(10000):
#     mne.viz.plot_topomap(evoked.data[:, i], evoked.info,show=False)
#     plt.savefig('D:\AI\BCI\dataset\BCI Competition IV\BCICIV_1_mat\\test\sub7/%d' %i)
#     plt.clf()
    

#ch_names =['Fp1', 'Fp2', 'F5', 'AFz', 'F6', 'T7', 'Cz', 'T8', 'P7', 'P3' , 'Pz', 'P4' , 'P8', 'O1', 'Oz', 'O2']
#sfreq = 512 #采样率
#info = mne.create_info(ch_names=ch_names, sfreq = sfreq,ch_types = 'eeg') #创建信号的信息
#info.set_montage('standard_1020')
#evoked = mne.EvokedArray(data, info)
#evoked.plot_sensors(ch_type='eeg')
#plt.show()
#
# for i in range(500):
#     mne.viz.plot_topomap(evoked.data[:, i], evoked.info,show=False)
#     plt.savefig('D:\AI\BCI\dataset\BCI Competition IV\BCICIV_1_mat\\test\sub1/%d' %i)
#     plt.clf()
# '''