#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, Input
from tensorflow import keras
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import scipy.io as sio
#%%
nSymbol = 5000						#用于训练的符号
cpNum = nSymbol // 4				#循环前缀
TotalSymbol = nSymbol + cpNum
M = 16								#调制阶数
SNR = list(range(0, 20))				#信噪比
batch_size = 256					
epochs = 20
#%%
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
#用不同的信噪比训练
train_label_index = data_send_index * len(SNR)   #标签索引
train_data_list = []
#%%
Es = np.linalg.norm(data_ofdm_send) ** 2 / TotalSymbol	#求每个符号的能量
Eb = Es / np.log2(M)							        #求每个比特的能量
Pe_simu = []

# 把不同信噪比下的接收信号加入到训练集中
for snrdB in SNR:
	snr = 10 ** (snrdB / 10.0)
	sigma = Eb / snr
	noise = np.sqrt(sigma/2) * np.random.randn(1, TotalSymbol) + \
	np.sqrt(sigma/2) * np.random.randn(1, TotalSymbol)*1j
	data_ofdm_receive = data_ofdm_send + noise
	data_fft = data_ofdm_receive[0, cpNum : cpNum+nSymbol]
	data_receive = np.fft.fft(data_fft)
	train_data_list += list(data_receive)
#%%
# ================
# 搭建网络
# ================
merged_inputs = Input(shape=(2,)) 
temp = layers.Dense(40,activation='relu')(merged_inputs)
temp = layers.BatchNormalization()(temp)
temp = layers.Dense(80, activation='relu')(temp)
temp = layers.BatchNormalization()(temp)
out= layers.Dense(M, activation='softmax')(temp)
model = models.Model(inputs=merged_inputs, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.summary()

#%%
train_label_tf = tf.one_hot(train_label_index, depth=M)       #编程one-hot类型数据
train_data_tmp = np.array(train_data_list)
data_real = np.real(train_data_tmp).reshape(-1, 1)         #拆成虚实, nSymbol*2列
data_imag = np.imag(train_data_tmp).reshape(-1, 1)

train_data = np.concatenate((data_real, data_imag), axis=-1)
train_label = np.array(train_label_tf)

# 打乱一下顺序，数据和标签要对应上
state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(state)
np.random.shuffle(train_label)

'''
path = './best_parameter.h5'
# 检测不行保存最好的
checkpointer = keras.callbacks.ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True, 
                                               save_weights_only=True)
# 调整学习率
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.333, patience=10, verbose=1, 
                                              mode='auto', min_delta=0.0001,min_lr=0.00001)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=20)
'''

history = model.fit(train_data, 
                    train_label, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=1, 
                    shuffle=True,
                    #分割一部分用于验证集
                    validation_split=0.2,
                    #callbacks=[checkpointer,early_stopping,reduce_lr]
					)

#%%
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 
acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy'] 
epochs = range(1, len(loss) + 1) 
 
plt.plot(epochs, acc, 'bo', label='Training accuracy') 
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation loss') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
# plt.show()
plt.savefig('Accuracy.pdf')
plt.savefig('Accuracy.svg')
#%%
test_origin_data = sio.loadmat('./Traditional_data.mat')
test_origin_data.keys()
#%%
test_data = test_origin_data['test_data']
test_label = test_origin_data['test_label']
Total = len(test_label[0])	#总符号个数
Pe_Tra_simu = test_origin_data['Pe_simu']
#%%
Pe_Deep_simu = []
# data_predict = model.predict(test_data, verbose=1)
for i in range(len(test_data)):
	data_predict_OneHot = model.predict(test_data[i])
	data_predict = np.argmax(data_predict_OneHot, axis=-1)
	Pe_Deep_simu.append((data_predict != test_label).sum() / Total)

#%%
snrdB = list(range(0, 20))
plt.semilogy(snrdB, Pe_Tra_simu[0], 'k-^')
plt.semilogy(snrdB, Pe_Deep_simu, 'r-.*')

plt.grid(True, which='major')
plt.grid(True, which='minor', linestyle='--')
plt.xlabel('SNR(dB)')
plt.ylabel('Symbol Error Rate')
plt.legend(['Traditional Method', 'Deep Learning'])
plt.axis([0, 18, 10**-3, 10**0])
plt.savefig('Deep_Tra_Compare.pdf')
plt.savefig('Deep_Tra_Compare.svg')
#%%
