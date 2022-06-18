import tensorflow as tf
import os
import datetime
from Seq2Seq import Encoder, Decoder
import tensorflow as tf


# 设置参数
data_path = './data/clean_data'  # 文件路径
epoch = 501  # 迭代训练次数
batch_size = 15  # 每批次样本数
embedding_dim = 256  # 词嵌入维度
hidden_dim = 512  # 隐层神经元个数
shuffle_buffer_size = 4  # 清洗数据集时将缓冲的实例数
device = -1  # 使用的设备ID，-1即不使用GPU
checkpoint_path = './data/model/'  # 模型参数保存的路径
MAX_LENGTH = 50  # 句子的最大词长
CONST = {'_BOS': 0, '_EOS': 1, '_PAD': 2, '_UNK': 3}# 最大输出句子的长度

# 加载词典
print(f'[{datetime.datetime.now()}] 加载词典...')
CONST = {'_BOS': 0, '_EOS': 1, '_PAD': 2, '_UNK': 3}
table = tf.lookup.StaticHashTable(
# 初始化后即不可变的通用哈希表。字典{键:值，键:值，键:值，键:值……} {"冷冻":0,"家里":1,……}
    initializer=tf.lookup.TextFileInitializer(
        os.path.join(data_path, 'all_dict.txt'),
        tf.string,
        tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64,
        tf.lookup.TextFileIndex.LINE_NUMBER
    ),  # 要使用的表初始化程序。有关支持的键和值类型，请参见HashTable内核。
    default_value=CONST['_UNK'] - len(CONST)  # 表中缺少键时使用的值。
)

# 建模
print(f'[{datetime.datetime.now()}] 创建一个seq2seq模型...')
encoder = Encoder(table.size().numpy() + len(CONST), embedding_dim, hidden_dim)
decoder = Decoder(table.size().numpy() + len(CONST), embedding_dim, hidden_dim)

# 设置优化器
print(f'[{datetime.datetime.now()}] 准备优化器...')
optimizer = tf.keras.optimizers.Adam()

# 设置模型保存
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

# 模型预测
def predict(sentence='你好'):
    # 导入训练参数
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
    # 给句子添加开始和结束标记
    sentence = '_BOS' + sentence + '_EOS'
    # 读取字段
    with open(os.path.join(data_path, 'all_dict.txt'), 'r', encoding='utf-8') as f:
        all_dict = f.read().split()
    # 构建: 词-->id的映射字典
    word2id = {j: i+len(CONST) for i, j in enumerate(all_dict)}
    word2id.update(CONST)
    # 构建: id-->词的映射字典
    id2word = dict(zip(word2id.values(), word2id.keys()))
    # 分词时保留_EOS 和 _BOS
    from jieba import lcut, add_word
    for i in ['_EOS', '_BOS']:
        add_word(i)
    # 添加识别不到的词，用_UNK表示
    inputs = [word2id.get(i, CONST['_UNK']) for i in lcut(sentence)]
    # 长度填充
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=MAX_LENGTH, padding='post', value=CONST['_PAD'])
    # 将数据转为tensorflow的数据类型
    inputs = tf.convert_to_tensor(inputs)
    # 空字符串，用于保留预测结果
    result = ''
    
    # 编码 
    enc_out, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([word2id['_BOS']], 0)

    for t in range(MAX_LENGTH):
        # 解码
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        # 预测出词语对应的id
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 通过字典的映射，用id寻找词，遇到_EOS停止输出
        if id2word.get(predicted_id, '_UNK') == '_EOS':
            break
        # 未预测出来的词用_UNK替代
        result += id2word.get(predicted_id, '_UNK')
        dec_input = tf.expand_dims([predicted_id], 0)

    return result # 返回预测结果

if __name__ == '__main__':
    while 1 :
        Q = input("Q: ")
        print("A: ", predict(Q))