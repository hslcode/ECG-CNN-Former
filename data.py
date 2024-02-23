import torch
import wfdb
import pywt
import numpy as np
import torch.utils.data as Data
from STFT import STFT
import matplotlib.pyplot as plt
import seaborn
from imblearn.over_sampling import SMOTE
# 测试集在数据集中所占的比例
RATIO = 0.3
# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata
# 读取心电数据和对应标签,并对数据进行小波去噪
def read_mit_bih(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('H:/post_design/ECG_C/datasheet/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)
    #rdata = data

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('H:/post_design/ECG_C/datasheet/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return
# 加载数据集并进行预处理
def loadData(re_loadData):
    if re_loadData:
        numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                     '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                     '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                     '231', '232', '233', '234']
        dataSet = []
        lableSet = []
        for n in numberSet:
            read_mit_bih(n, dataSet, lableSet)
        # 转numpy数组,打乱顺序
        dataSet = np.array(dataSet).reshape(-1, 300)
        lableSet = np.array(lableSet).reshape(-1, 1)
        train_ds = np.hstack((dataSet, lableSet))
        np.random.shuffle(train_ds)

        # 数据集及其标签集
        X = train_ds[:, :300].reshape(-1, 300, 1)
        Y = train_ds[:, 300]
        # 测试集及其标签集
        shuffle_index = np.random.permutation(len(X))
        test_length = int(RATIO * len(shuffle_index))
        test_index = shuffle_index[:np.floor(0.8*test_length).astype(np.int16)]
        val_index = shuffle_index[np.floor(0.8*test_length).astype(np.int16)+1:test_length]
        train_index = shuffle_index[test_length:]

        X_test, Y_test = X[test_index], Y[test_index].astype(np.int8)
        X_val, Y_val = X[val_index], Y[val_index].astype(np.int8)
        X_train, Y_train = X[train_index], Y[train_index].astype(np.int8)
        #训练集数据均衡-基于SMOTE算法
        over_samples = SMOTE(random_state=0)
        over_samples_X_train, over_samples_Y_train = over_samples.fit_resample(np.squeeze(X_train, axis=2), Y_train)
        X_train = np.expand_dims(over_samples_X_train, axis=2)
        Y_train = over_samples_Y_train

        np.save("X_train.npy",X_train)
        np.save("Y_train.npy", Y_train)
        np.save("X_test.npy", X_test)
        np.save("Y_test.npy", Y_test)
        np.save("X_val.npy", X_val)
        np.save("Y_val.npy", Y_val)
    else:
        X_train = np.load("X_train.npy")
        Y_train = np.load("Y_train.npy")
        X_test = np.load("X_test.npy")
        Y_test = np.load("Y_test.npy")
        X_val = np.load("X_val.npy")
        Y_val = np.load("Y_val.npy")
    # '''
    #统计各类别数量
    statistic_tarin = np.zeros(5)
    for index in range(len(Y_train[:])):
        statistic_tarin[Y_train[index]] +=1
    print("the train data class numbers:", statistic_tarin)

    statistic_test = np.zeros(5)
    for index in range(len(Y_test[:])):
        statistic_test[Y_test[index]] +=1
    print("the test data class numbers:", statistic_test)

    statistic_val = np.zeros(5)
    for index in range(len(Y_val[:])):
        statistic_val[Y_val[index]] +=1
    print("the val data class numbers:", statistic_val)
    # '''
    #return X_train, Y_train.astype(np.int8), X_test, Y_test.astype(np.int8)
    return X_train, Y_train, X_test, Y_test,X_val, Y_val

def make_data_stft(Signals,fs,n,win = 'hann'):
    stft = STFT(fs, n, win)
    Z_data = stft.stft_once(Signals[0, :, 0])
    for i in range(len(Signals[:,1,0])):
        print("正在处理第%d",i)
        F,T,Z = stft.stft_once(Signals[i,:,0])
        Z = np.transpose(Z, (1, 0)) #[fre,time]->[time,fre]
        Z = np.expand_dims(Z, axis=0)
        if i==0:
            Z_data = Z
        else:
            Z_data = np.append(Z_data, Z, axis=0)
        pass
    return torch.FloatTensor(Z_data)
# 混淆矩阵
def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    # 归一化
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # 绘图
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
def get_data():
    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")
    X_test = np.load("X_test.npy")
    Y_test = np.load("Y_test.npy")

    X_train = np.squeeze(X_train,axis=2)
    X_test = np.squeeze(X_test, axis=2)

    X_train = np.expand_dims(X_train,axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    #统计训练集各类别数量
    statistic_train = np.zeros(5)
    for index in range(len(Y_train)):
        statistic_train[Y_train[index]] += 1
    print("train lebals number:",statistic_train)
    #统计训练集各类别数量
    statistic_test = np.zeros(5)
    for index in range(len(Y_test)):
        statistic_test[Y_test[index]] += 1
    print("train lebals number:",statistic_test)
    return X_train,Y_train,X_test,Y_test
class MyDataSet(Data.Dataset):
    """自定义DataLoader"""
    def __init__(self, Z_datas1,Z_datas2, Lables):
        super(MyDataSet, self).__init__()
        self.Z_datas1 = Z_datas1
        self.Z_datas2 = Z_datas2
        self.Lables = Lables

    def __len__(self):
        return self.Z_datas1.shape[0]

    def __getitem__(self, idx):
        return self.Z_datas1[idx], self.Z_datas2[idx],self.Lables[idx]