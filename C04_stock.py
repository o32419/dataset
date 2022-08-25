import os,math
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
from sklearn.preprocessing   import MinMaxScaler
from sklearn                 import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data = pd.read_csv('D:/BaiduNetdiskDownload/SH600519.csv')  # 读取股票文件
#print(data)
#取出第三列
training_set = data.iloc[0:2426 - 300, 2:3].values
test_set = data.iloc[2426 - 300:, 2:3].values
#归一化
sc= MinMaxScaler(feature_range=(0, 1))
training_set = sc.fit_transform(training_set)
test_set= sc.transform(test_set)
#设置测试集、训练集
x_train = []
y_train = []

x_test = []
y_test = []

"""
使用前60天的开盘价作为输入特征x_train
    第61天的开盘价作为输入标签y_train

for循环共构建2426-300-60=2066组训练数据。
       共构建300-60=260组测试数据
"""
for i in range(60, len(training_set)):
    x_train.append(training_set[i - 60:i, 0])
    y_train.append(training_set[i, 0])

for i in range(60, len(test_set)):
    x_test.append(test_set[i - 60:i, 0])
    y_test.append(test_set[i, 0])

# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
#将训练数据调整为数组（array）
x_train, y_train = np.array(x_train), np.array(y_train)
x_test,  y_test  = np.array(x_test),  np.array(y_test)
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
x_test  = np.reshape(x_test,  (x_test.shape[0], 60, 1))

model = tf.keras.Sequential([
    SimpleRNN(100, return_sequences=True), #布尔值。是返回输出序列中的最后一个输出，还是全部序列。
    Dropout(0.1),                         #防止过拟合
    SimpleRNN(100),
    Dropout(0.1),
    Dense(1)
])
# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方误差
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=20,
                    validation_data=(x_test, y_test),
                    validation_freq=1)                  #测试的epoch间隔数

model.summary()
# plt.plot(history.history['loss']    , label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss by K同学啊')
# plt.legend()
# plt.show()
##预测
predicted_stock_price = model.predict(x_test)                       # 测试集输入模型进行预测
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # 对预测数据还原---从（0，1）反归一化到原始范围
real_stock_price = sc.inverse_transform(test_set[60:])              # 对真实数据还原---从（0，1）反归一化到原始范围

# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction by K同学啊')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
#评估
MSE   = metrics.mean_squared_error(predicted_stock_price, real_stock_price)
RMSE  = metrics.mean_squared_error(predicted_stock_price, real_stock_price)**0.5
MAE   = metrics.mean_absolute_error(predicted_stock_price, real_stock_price)
R2    = metrics.r2_score(predicted_stock_price, real_stock_price)

print('均方误差: %.5f' % MSE)
print('均方根误差: %.5f' % RMSE)
print('平均绝对误差: %.5f' % MAE)
print('R2: %.5f' % R2)

