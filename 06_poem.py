##相同作品（荷马的伊利亚特）三个不同版本的英文翻译
#TextLineDataset 通常被用来以文本文件构建数据集（原文件中的一行为一个样本) 。
# 这适用于大多数的基于行的文本数据（例如，诗歌或错误日志) 。
#pip install  tfds-nightly==3.2.1.dev202009080105 -i https://pypi.tuna.tsinghua.edu.cn/simple
import tensorflow as tf
import tensorflow_datasets as tfds
import os

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
  text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)
parent_dir = os.path.dirname(text_dir)
#print(parent_dir)
#将整个文件加载到自己的数据集中
def labeler(example, index):
  return example, tf.cast(index, tf.int64)

labeled_data_sets = []
for i, file_name in enumerate(FILE_NAMES):
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)
#将这些标记的数据集合并到一个数据集中，然后对其进行随机化操作。
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000
all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)
#查看
# for ex in all_labeled_data.take(5):
#   print(ex)
##构建词汇表
tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)
vocab_size = len(vocabulary_set)
#print(vocab_size)
#构建编码器
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
example_text = next(iter(all_labeled_data))[0].numpy()
#print(example_text)
encoded_example = encoder.encode(example_text)
#print(encoded_example)
#在数据集上运行编码器
def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode,
                                       inp=[text, label],
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually:
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label
all_encoded_data = all_labeled_data.map(encode_map_fn)
#分割数据集
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)
#输出样式
# sample_text, sample_labels = next(iter(test_data))
# print(sample_text[0], sample_labels[0])
#由于我们引入了一个新的 token 来编码（填充零），因此词汇表大小增加了一个。
vocab_size += 1
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# 一个或多个紧密连接的层
# 编辑 `for` 行的列表去检测层的大小
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))
# 输出层。第一个参数是标签个数。
model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, epochs=3, validation_data=test_data)
eval_loss, eval_acc = model.evaluate(test_data)
print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))