import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt

import IPython.display as display

AUTOTUNE = tf.data.experimental.AUTOTUNE
data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
#print(data_root)
##   遍历该压缩文件下的文件和目录
#for item in data_root.iterdir():
#  print(item)
##  获取图片总数
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
print(image_count)
print(all_image_paths[:10])
##显示几张图片
attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)


def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel).replace("\\","/")].split(' - ')[:-1])

for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print(caption_image(image_path))
  print()
#显示可用的标签
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
#print(label_names)
#为标签分配索引
label_to_index = dict((name, index) for index, name in enumerate(label_names))
#print(label_to_index)
#为所有图片添加索引
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
print("First 10 labels indices: ", all_image_labels[:10])
#读取一张图片
img_path = all_image_paths[0]
img_raw = tf.io.read_file(img_path)
#print(repr(img_raw)[:100]+"...")
#解码为tensor张量
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape,img_tensor.dtype)
#根据你的模型调整其大小：
img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())
##将这些封装
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range
  return image
def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)
##显示第0张图片
# image_path = all_image_paths[0]
# label = all_image_labels[0]
# plt.imshow(load_and_preprocess_image(img_path))
# plt.grid(False)
# plt.xlabel(caption_image(img_path))
# plt.title(label_names[label].title())
# plt.show()
# print()
##将字符串数组切片，得到一个字符串数据集
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
##显示处理后的图片
# plt.figure(figsize=(8,8))
# for n, image in enumerate(image_ds.take(4)):
#   plt.subplot(2,2,n+1)
#   plt.imshow(image)
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#   plt.xlabel(caption_image(all_image_paths[n]))
#   plt.show()
##构造标签数据集
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
#由于这些数据集顺序相同，你可以将他们打包在一起得到一个(图片, 标签)对数据集：
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
#print(image_label_ds)
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
###打乱顺序(1)
BATCH_SIZE = 32
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
# ds = image_label_ds.shuffle(buffer_size=image_count)
# ds = ds.repeat()
# ds = ds.batch(BATCH_SIZE)
# # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
# ds = ds.prefetch(buffer_size=AUTOTUNE)
###打乱顺序(2)
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
##该模型副本会被用于一个简单的迁移学习例子,该模型期望它的输出被标准化至 [-1,1] 范围内
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False
#在你将输出传递给 MobilNet 模型之前，你需要将其范围从 [0,1] 转化为 [-1,1]
def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)
#传递一个batch给mobilenet,MobileNet 为每张图片的特征返回一个 6x6 的空间网格
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names), activation = 'softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
model.summary()
##计算需要运行的步数
steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
print(steps_per_epoch)
#出于演示，只运行3步
model.fit(ds, epochs=1, steps_per_epoch=3)

