from tensorflow import keras


def text_cnn(seq_length, vocab_size, embedding_dim, num_cla, kernelNum):
    """
    :param seq_length:  输入的文字序列长度
    :param vocab_size:  词汇库的大小
    :param embedding_dim:  生成词向量的特征维度
    :param num_cla: 分类类别
    :return: keras model
    """
    inputX = keras.layers.Input(shape=(seq_length,), dtype='int32')
    embOut = keras.layers.Embedding(vocab_size, embedding_dim, input_length=seq_length)(inputX)
    # 分别使用长度为3,4,5的词窗去执行卷积
    conv1 = keras.layers.Conv1D(kernelNum, 3, padding='valid', strides=1, activation='relu')(embOut)
    maxp1 = keras.layers.MaxPool1D(pool_size=int(conv1.shape[1]))(conv1)

    conv2 = keras.layers.Conv1D(kernelNum, 4, padding='valid', strides=1, activation='relu')(embOut)
    maxp2 = keras.layers.MaxPool1D(pool_size=int(conv2.shape[1]))(conv2)

    conv3 = keras.layers.Conv1D(kernelNum, 5, padding='valid', strides=1, activation='relu')(embOut)
    maxp3 = keras.layers.MaxPool1D(pool_size=int(conv3.shape[1]))(conv3)

    # 合并三个模型的输出向量
    cnn = keras.layers.Concatenate(axis=-1)([maxp1, maxp2, maxp3])
    flat = keras.layers.Flatten()(cnn)
    dense1 = keras.layers.Dense(128)(flat)
    drop = keras.layers.Dropout(0.25)(dense1)
    denseRelu = keras.layers.ReLU()(drop)
    predictY = keras.layers.Dense(num_cla, activation='softmax')(denseRelu)
    # 编译模型
    model = keras.models.Model(inputs=inputX, outputs=predictY)
    # 指定loss的计算方法，设置优化器，编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def text_cnn_V2(seq_length, vocab_size, embedding_dim, num_cla, kernelNum=128):
    """
    :param seq_length:  输入的文字序列长度
    :param vocab_size:  词汇库的大小
    :param embedding_dim:  生成词向量的特征维度
    :param num_cla: 分类类别
    :return: keras model
    """
    inputX = keras.layers.Input(shape=(seq_length,), dtype='int32')
    embOut = keras.layers.Embedding(vocab_size, embedding_dim, input_length=seq_length)(inputX)
    # 分别使用长度为3,4,5的词窗去执行卷积
    conv1 = keras.layers.Conv1D(kernelNum, 3, padding='valid', strides=1, activation='relu')(embOut)
    maxp1 = keras.layers.SeparableConv1D(filters=int(conv1.shape[2]), kernel_size=int(conv1.shape[1]))(conv1)

    conv2 = keras.layers.Conv1D(kernelNum, 4, padding='valid', strides=1, activation='relu')(embOut)
    maxp2 = keras.layers.SeparableConv1D(filters=int(conv2.shape[2]), kernel_size=int(conv2.shape[1]))(conv2)

    conv3 = keras.layers.Conv1D(kernelNum, 5, padding='valid', strides=1, activation='relu')(embOut)
    maxp3 = keras.layers.SeparableConv1D(filters=int(conv3.shape[2]), kernel_size=int(conv3.shape[1]))(conv3)

    # 合并三个模型的输出向量
    cnn = keras.layers.Concatenate(axis=2)([maxp1, maxp2, maxp3])
    bn = keras.layers.BatchNormalization()(cnn)
    conv4 = keras.layers.Conv1D(num_cla, kernel_size=int(cnn.shape[1]), activation='softmax')(bn)
    # predictY = keras.layers.Lambda(keras.backend.squeeze, arguments={'axis': 1})(conv4)
    predictY = keras.layers.Flatten()(conv4)
    # 编译模型
    model = keras.models.Model(inputs=inputX, outputs=predictY)
    # 指定loss的计算方法，设置优化器，编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def simpleNet(seq_length, vocab_size, embedding_dim, num_cla, kernelNum=128):
    inputX = keras.layers.Input(shape=(seq_length,), dtype='int32')
    embOut = keras.layers.Embedding(vocab_size, embedding_dim, input_length=seq_length)(inputX)
    # 使用长度为5的词窗去执行卷积
    conv1 = keras.layers.Conv1D(kernelNum, 5, padding='same', strides=1)(embOut)
    maxp1 = keras.layers.MaxPool1D(pool_size=int(conv1.shape[1]))(conv1)
    flat = keras.layers.Flatten()(maxp1)
    dense1 = keras.layers.Dense(128)(flat)
    drop = keras.layers.Dropout(0.25)(dense1)
    denseRelu = keras.layers.ReLU()(drop)
    predictY = keras.layers.Dense(num_cla, activation='softmax')(denseRelu)
    # 编译模型
    model = keras.models.Model(inputs=inputX, outputs=predictY)
    # 指定loss的计算方法，设置优化器，编译模型
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
