# Fashion MNIST 数据集找到最佳超参数;Keras Tuner 是一个库，可帮助您为 TensorFlow 程序选择最佳的超参数集。
#超参数有两种类型：
#模型超参数：影响模型的选择，例如隐藏层的数量和宽度
#算法超参数：影响学习算法的速度和质量，例如随机梯度下降 (SGD) 的学习率以及 k 近邻 (KNN) 分类器的近邻数

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0
#定义模型,定义超参数搜索空间,超模型
def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model
#实例化调节器并执行超调
#Keras Tuner 提供了四种调节器：RandomSearch、Hyperband、BayesianOptimization 和 Sklearn
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')
#创建回调以在验证损失达到特定值后提前停止训练。
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#运行超参数搜索,搜索方法的参数也与 tf.keras.model.fit 所用参数相同
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
#使用从搜索中获得的超参数找到训练模型的最佳周期数
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
#重新实例化超模型并使用上面的最佳周期数对其进行训练。
hypermodel = tuner.hypermodel.build(best_hps)
# Retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)