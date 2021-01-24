import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt

train_size = 2187  # 训练集数据个数,测试集第一个数据的下标,2187为10~18年的数据总长度
time_stamp = 54  # 时间步,最大为58
time_stamp_ago = time_stamp - 1  # 表示前time_stamp_ago天

data = pd.read_csv("quarter_train_data.csv", encoding='gbk').drop('交易日期', axis=1).values
ShenZhen20 = pd.read_csv("沪深20指数-训练集.csv", encoding='gbk').drop('交易日期', axis=1).values

sc1 = StandardScaler()
sc2 = StandardScaler()
data = sc1.fit_transform(data)
ShenZhen20 = sc2.fit_transform(ShenZhen20)

# 构建训练集
X_train = data[0:train_size]

tmp = []
for i in range(57, train_size):  # 57是2010/03/31的下标
    tmp.append(X_train[i - time_stamp_ago:i + 1])  # 前time_stamp_ago天和今天作为一组输入
X_train = tmp
Y_train = ShenZhen20[0:len(X_train)]
X_train, Y_train = np.array(X_train), np.array(Y_train)

# 构建RRN神经网络（LSTM）
model = keras.Sequential([
    keras.layers.LSTM(units=100, activation='tanh', return_sequences=True),  # 隐层
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=60),  # 隐层
    keras.layers.Dense(units=1)  # 输出层
])

# 模型编译 RMsProp或adam两个优化器可选，loss选mse或mae，metrics必须是mae
model.compile(optimizer='RMsProp', loss='mse', metrics=['mae'])

# 输入数据
history = model.fit(X_train, Y_train, epochs=40, verbose=1)

# 预测2019年数据
X_predict = data[2187 - time_stamp_ago:2431]

# 将数据处理成矩阵形式
tmp = []
for i in range(time_stamp_ago, len(X_predict)):
    tmp.append(X_predict[i - time_stamp_ago:i + 1])
X_predict = np.array(tmp)

# 预测19年股票指数
Y_predict = model.predict(X_predict)
Y_predict = sc2.inverse_transform(Y_predict)

# 构建测试集
data = data[57 - time_stamp_ago:2187]
sz = ShenZhen20
# 将数据处理成矩阵形式
x_tmp = []
for i in range(time_stamp_ago, len(data)):
    x_tmp.append(data[i - time_stamp_ago:i + 1])
data = np.array(x_tmp)
# 测试集数据预测
test_pre = model.predict(data)
# 反归一化预测值
test_pre = sc2.inverse_transform(test_pre)
# 反归一化真实值
sz = sc2.inverse_transform(sz)
# 输出 MAE 分数
print(sz.shape)
print(test_pre.shape)



for unit in Y_predict:
    print(unit)


loss = sum(abs(sz.reshape(len(test_pre), 1) - test_pre)) / len(test_pre)
print(f"MAE分数{loss}")

# 模型保存
print("Saving model to disk \n")
model.save("model_by_day.h5")

ob = plt.figure(figsize=(25, 10))
ob.patch.set_facecolor('gray')
plt.style.use('ggplot')
plt.title("Quarter_2019")
real = sc2.inverse_transform(ShenZhen20[0:2130])
pre = Y_predict.reshape(-1, 1)
x = range(len(real), len(real) + len(pre))
plt.scatter(x, pre, s=2, c='g', label='19_predict')
plt.scatter(range(len(real)), real, s=2, c='b', label='10~18_real')
plt.plot(range(len(test_pre.reshape(1, -1)[0])), test_pre.reshape(1, -1)[0], 'r', label='10~18_predict')
plt.legend(loc='upper left')
plt.show()

