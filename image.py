import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# #检查训练集中的第一个图像
# plt.figure() #生成一个画框
# plt.imshow(train_images[0]) #展示一副热度图
# plt.colorbar()  #显示色彩范围
# plt.grid(False)  #设置图表中的网格线
# plt.show()
train_images = train_images / 255.0
test_images = test_images / 255.0
# #展示前25幅图片
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001),activation='relu'),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc) #测试集误差大于训练集，过拟合，添加L2正则化
#附加一个 softmax 层，将 logits 转换成更容易理解的概率
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])#输出10个概率值
print('查看第0个预测值：',np.argmax(predictions[0]))#返回最大值对应的索引
print('真实值：',test_labels[0])
#取出一个测试集图片验证模型
img = test_images[1]
img = (np.expand_dims(img,0)) #分批次读取图片，格式[batch,height,width,channel]
predictions_single = probability_model.predict(img)
print(np.argmax(predictions_single))
print('真实值：',test_labels[1])