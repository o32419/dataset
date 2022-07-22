import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import numpy as np

imdb = keras.datasets.imdb
#num_words 定义的是大于该词频的单词会被读取。如果单词的词频小于该整数，会用oov_char定义的数字代替。默认是用2代替。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# print("Training entries: {}, labels: {}".format(len(train_data), len(test_labels)))
# print(train_data[0])
# print(len(train_data[0]), len(train_data[1]))#第一条、第二条单词数量不同

####将整数转换成单词
# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()
# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
#print(decode_review(train_data[0]))
#填充、去掉整数
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
#print(train_data[0])

vocab_size = 10000
model = keras.Sequential()
#layers.Embedding(input_dim=**, output_dim=, input_length=): 输出：(batch_size,input_length,output_dim)
#输入整数最大不超过input_dim
#使用矩阵相乘，将稀疏矩阵降维，节省存储空间
#Embedding层本质也是一个映射，映射为一个指定维度的向量，该向量是一个变量，通过学习寻找到最优值
model.add(keras.layers.Embedding(vocab_size, 16))
#输出 (batch_size, features)，features表示一行文本使用多少个维度(定长输出向量)进行表示
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#从训练集上划分验证集（为什么现在不使用测试集？我们的目标是只使用训练数据来开发和调整模型，然后只使用一次测试数据来评估准确率（accuracy））
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
#validation_data用于制定测试集
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)
#model.fit() 返回一个 History 对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件
history_dict = history.history
print(history_dict.keys())
#绘制训练与验证过程的损失值（loss）和准确率（accuracy）
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

