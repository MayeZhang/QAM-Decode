#%%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import scipy.io as sio
#%%
nSymbol = 10000
cpNum = nSymbol // 4
TotalSymbol = nSymbol + cpNum
SNR = list(range(0, 20))
M = 16

# 生成发送数据
data_scource = np.random.randint(0, 2, [nSymbol, 4])

# 映射成16QAM符号
mapBitToSymbol = {
	(0, 0, 0, 0) : [-3+3*1j, 0],
	(0, 0, 0, 1) : [-3+1*1j, 1],
	(0, 0, 1, 0) : [-3-3*1j, 2],
	(0, 0, 1, 1) : [-3-1*1j, 3],
	(0, 1, 0, 0) : [-1+3*1j, 4],
	(0, 1, 0, 1) : [-1+1*1j, 5],
	(0, 1, 1, 0) : [-1-3*1j, 6],
	(0, 1, 1, 1) : [-1-1*1j, 7],
	(1, 0, 0, 0) : [3+3*1j, 8],
	(1, 0, 0, 1) : [3+1*1j, 9],
	(1, 0, 1, 0) : [3-3*1j, 10],
	(1, 0, 1, 1) : [3-1*1j, 11],
	(1, 1, 0, 0) : [1+3*1j, 12],
	(1, 1, 0, 1) : [1+1*1j, 13],
	(1, 1, 1, 0) : [1-3*1j, 14],
	(1, 1, 1, 1) : [1-1*1j, 15],
}

data_send = []
data_send_index = []
for i in range(nSymbol):
	data_send.append(mapBitToSymbol[tuple(data_scource[i])][0]) 	   #调制后的符号
	data_send_index.append(mapBitToSymbol[tuple(data_scource[i])][1])  #符号索引

data_ifft = np.fft.ifft(data_send)		#变换到时域
data_ofdm_send = np.hstack([data_ifft[-cpNum:], data_ifft])
#%%
Es = np.linalg.norm(data_ofdm_send) ** 2 / TotalSymbol	#求每个符号的能量
Eb = Es / np.log2(M)							        #求每个比特的能量
Pe_simu = []
test_data = []
test_label = data_send_index

for snrdB in SNR:
    snr = 10 ** (snrdB / 10.0)
    sigma = Eb / snr
    noise = np.sqrt(sigma/2) * np.random.randn(1, TotalSymbol) + \
    np.sqrt(sigma/2) * np.random.randn(1, TotalSymbol)*1j
    data_ofdm_receive = data_ofdm_send + noise
    data_fft = data_ofdm_receive[0, cpNum : cpNum+nSymbol]
    data_receive = np.fft.fft(data_fft)
    test_data.append(np.concatenate((np.real(data_receive).reshape(-1, 1), 
                                     np.imag(data_receive.reshape(-1, 1))), axis=-1))
	

    TotalErrorSymbol = 0
    #data_receive_index = []

    for i in range(len(data_receive)):
        min_index = 0
        min_value = np.linalg.norm(mapBitToSymbol[(0, 0, 0, 0)][0] - data_receive[i])
        
        for bit, (symbol, index) in mapBitToSymbol.items():
            error_value = np.linalg.norm(symbol - data_receive[i])
            if error_value < min_value:
                min_index = index
                min_value = error_value
        
        # 统计错误符号个数
        if min_index != data_send_index[i]:
            TotalErrorSymbol += 1
    #print(TotalErrorSymbol)
    Pe_simu.append(TotalErrorSymbol / nSymbol)

#%%
import math
def Q(x):
    return math.erfc(x / math.sqrt(2)) / 2
a= 4 * (1 - 1/math.sqrt(M)) / math.log2(M)
k = math.log2(M)
b = 3 * k / (M-1)
Pe_theory = []

# 计算理论误码率
for snrdB in range(20):
    Pe_theory.append(a * Q(math.sqrt(b*10**(snrdB/10))) * math.log2(M))

# 绘图
snrdB = list(range(0, 20))
plt.semilogy(snrdB, Pe_theory, 'r-.*')
plt.semilogy(snrdB, Pe_simu, 'k-^')

plt.grid(True, which='major')
plt.grid(True, which='minor', linestyle='--')
plt.xlabel('SNR(dB)')
plt.ylabel('Symbol Error Rate')
plt.legend(['Theory', 'Simulation'])
plt.axis([0, 18, 10**-3, 10**0])
plt.savefig('Theory_Tra_Compare.svg')
plt.savefig('Theory_Tra_Compare.pdf')
#%%
# 保存数据
sio.savemat('./Traditional_data.mat', 
            {'test_data':test_data,
             'test_label':test_label,
             'Pe_simu':Pe_simu}
            )

