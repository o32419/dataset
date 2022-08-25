import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255.0, test_images / 255.0
print(train_images.shape,test_images.shape,train_labels.shape,test_labels.shape)
#调整数据到我们需要的格式
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
print(train_images.shape,test_images.shape,train_labels.shape,test_labels.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure(figsize=(20,10))
# for i in range(20):
#     plt.subplot(5,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# #plt.show()

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 卷积层
    layers.MaxPooling2D((2, 2)),  # 池化层
    layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层
    layers.MaxPooling2D((2, 2)),  # 池化层2，2*2采样
    layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层

    layers.Flatten(),  # Flatten层
    layers.Dense(64, activation='relu'),  # 全连接层
    layers.Dense(10)  # 输出层
])

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='accuracy')
plt.plot(epochs,val_acc, label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_loss,test_acc)
#测试
print('选择一个样本：',class_names[test_labels[5]])

pre = model.predict(test_images)
print('预测值：',class_names[np.argmax(pre[5])])

