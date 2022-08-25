import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255.0, test_images / 255.0
print(train_images.shape,test_images.shape,train_labels.shape,test_labels.shape)
plt.figure(figsize=(20,10))
for i in range(20):
    plt.subplot(5,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
#plt.show()

#调整数据到我们需要的格式
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

print(train_images.shape,test_images.shape,train_labels.shape,test_labels.shape)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 卷积层
    layers.MaxPooling2D((2, 2)),  # 池化层1
    layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层
    layers.MaxPooling2D((2, 2)),  # 池化层

    layers.Flatten(),  # Flatten层
    layers.Dense(64, activation='relu'),  # 全连接层
    layers.Dense(10)  # 输出层
])
# 打印网络结构
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
test_loss, test_acc=model.evaluate(test_images,  test_labels)
print(test_loss,test_acc)
#附加一个 softmax 层，将 logits 转换成更容易理解的概率
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[5])#输出10个概率值
print('查看第0个预测值：',np.argmax(predictions[5]))#返回最大值对应的索引
print('真实值：',test_labels[5])





