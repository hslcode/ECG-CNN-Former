import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import seaborn
import pywt
class STFT:
    def __init__(self,Fs,n,win = 'hann'):
        self.data = []
        self.Fs = Fs
        self.window = win
        self.n = n
        self.f = 0
        self.t = 0
        self.z = 0
    def stft_once(self,data):
        self.data = data
        # STFT
        self.f, self.t, self.z = stft(self.data, fs=self.Fs, window=self.window, nperseg=self.n, noverlap=(int)(self.n*0.9),return_onesided=True)
        index = np.where(self.f[:]>100)
        self.z = np.delete(self.z,index,axis=0)
        self.f = np.delete(self.f, index, axis=0)
        # 求幅值
        self.z = np.abs(self.z)
        # 标准化
        self.z = (self.z - np.mean(self.z)) / np.std(self.z)
        # self.z = (self.z - np.mean(self.z)) / np.std(self.z)
        # 转置
        # self.z = np.transpose(self.z,(1,0))
        return self.f, self.t, self.z
    def stft_show(self):
        t = np.linspace(1, len(self.data), len(self.data))
        plt.plot(t,self.data, 'b')
        plt.xlabel('pinlv(hz)')
        plt.ylabel('amplitude')
        plt.title("pinputu")
        plt.show()
        # 如下图所示
        plt.pcolormesh(self.t, self.f, self.z, vmin=0, vmax=self.z.max(), shading='gouraud')
        plt.colorbar()
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.tight_layout()
        plt.show()
        print(np.shape(self.z))
        print(np.shape(self.t))
        print(np.shape(self.f))
        print(self.z)
        seaborn.set_theme()
        seaborn.heatmap(self.z,vmin=0, vmax=self.z.max())
        plt.show()
def cwt(data,fs,n):
    #t = np.arange(0, 1.0, 1.0 / fs)  # 0-1.0之间的数，步长为1.0/sampling_rate
    t = np.linspace(1, len(data), len(data))
    wavename = "cgau8"  # 小波函数
    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * n  # 常数c
    scales = cparam / np.arange(n, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    cwtmatr, cwt_fre = pywt.cwt(data, scales, wavename, 1.0 / fs)  # 连续小波变换模块
    cwtmatr = np.abs(cwtmatr)
    return cwt_fre,t,cwtmatr
