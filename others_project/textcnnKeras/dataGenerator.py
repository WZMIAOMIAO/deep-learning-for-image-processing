from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random


def content2idList(content, word2id_dict):
    """
    该函数的目的是将文本转换为对应的汉字数字id
    content：输入的文本
    word2id_dict：用于查找转换的字典
    """
    idList = []
    for word in content:  # 遍历每一个汉字
        if word in word2id_dict:  # 当刚文字在字典中时才进行转换，否则丢弃
            idList.append(word2id_dict[word])
    return idList


def generatorInfo(batch_size, seq_length, num_classes, file_name):
    """
    batch_size：生成数据的batch size
    seq_length：输入文字序列长度
    num_classes：文本的类别数
    file_name：读取文件的路径
    """
    # 读取词库文件
    with open('./cnews/cnews.vocab.txt', encoding='utf-8') as file:
        vocabulary_list = [k.strip() for k in file.readlines()]
    word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])

    # 读取文本文件
    with open(file_name, encoding='utf-8') as file:
        line_list = [k.strip() for k in file.readlines()]
        data_label_list = []  # 创建数据标签文件
        data_content_list = []  # 创建数据文本文件
        for k in line_list:
            t = k.split(maxsplit=1)
            data_label_list.append(t[0])
            data_content_list.append(t[1])

    data_id_list = [content2idList(content, word2id_dict) for content in data_content_list]  # 将文本数据转换拿为数字序列
    # 将list数据类型转换为ndarray数据类型，并按照seq_length长度去统一化文本序列长度，
    # 若长度超过设定值将其截断保留后半部分，若长度不足前面补0
    data_X = keras.preprocessing.sequence.pad_sequences(data_id_list, seq_length, truncating='pre')
    labelEncoder = LabelEncoder()
    data_y = labelEncoder.fit_transform(data_label_list)  # 将文字标签转为数字标签
    data_Y = keras.utils.to_categorical(data_y, num_classes)  # 将数字标签转为one-hot标签

    while True:
        selected_index = random.sample(list(range(len(data_y))), k=batch_size)  # 按照数据集合的长度随机抽取batch_size个数据的index
        batch_X = data_X[selected_index]  # 随机抽取的文本信息（数字化序列）
        batch_Y = data_Y[selected_index]  # 随机抽取的标签信息（one-hot编码）
        yield (batch_X, batch_Y)

