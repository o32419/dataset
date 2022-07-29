#这种情况下你会希望分割图像，也就是给图像中的每个像素各分配一个标签。
#本教程将使用的数据集是 Oxford-IIIT Pet 数据集，由 Parkhi et al. 创建。该数据集由图像、图像所对应的标签、以及对像素逐一标记的掩码组成。掩码其实就是给每个像素的标签。每个像素分别属于以下三个类别中的一个：

#类别 1：像素是宠物的一部分。
#类别 2：像素是宠物的轮廓。
#类别 3：以上都不是/外围像素。
import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

#这个数据集已经集成在 Tensorflow datasets 中，只需下载即可。
# 图像分割掩码在版本 3.0.0 中才被加入，因此我们特别选用这个版本。
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
#下面的代码进行了一个简单的图像翻转扩充。然后，将图像标准化到 [0,1]。
# 最后，如上文提到的，像素点在图像分割掩码中被标记为 {1, 2, 3} 中的一个。
# 为了方便起见，我们将分割掩码都减 1，得到了以下的标签：{0, 1, 2}。
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask
#数据集已经包含了所需的测试集和训练集划分，所以我们也延续使用相同的划分
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)
#我们来看一下数据集中的一例图像以及它所对应的掩码
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])
