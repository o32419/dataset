import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


#tfds.load 下载数据并将其存储为 tfrecord 文件。加载 tfrecord 并创建 tf.data.Dataset。
#tfrecord文件测试集训练集在一起
#[:60%]  训练集的前60%
# 25,000 条用于训练，另外 25,000 条用于测试;训练集再分为60%训练样本（15000），验证样本40%（10000）
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)
#显示前10个数据
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
#print(train_examples_batch)
embedding = "https://hub.tensorflow.google.cn/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
#print(hub_layer(train_examples_batch[:3]))
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)
results = model.evaluate(test_data.batch(512), verbose=2)
#model.metrics_names 将提供标量输出的显示标签
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))