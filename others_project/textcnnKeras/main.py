from models import text_cnn, simpleNet, text_cnn_V2
from dataGenerator import generatorInfo
from tensorflow import keras

vocab_size = 5000
seq_length = 600
embedding_dim = 64
num_classes = 10
trainBatchSize = 64
evalBatchSize = 200
steps_per_epoch = 50000 // trainBatchSize
epoch = 2
workers = 4
logdir = './log/'
trainFileName = './cnews/cnews.train.txt'
evalFileName = './cnews/cnews.test.txt'

model = text_cnn(seq_length=seq_length,
                 vocab_size=vocab_size,
                 embedding_dim=embedding_dim,
                 num_cla=num_classes,
                 kernelNum=64)

trainGenerator = generatorInfo(trainBatchSize, seq_length, num_classes, trainFileName)
evalGenerator = generatorInfo(evalBatchSize, seq_length, num_classes, evalFileName)


def lrSchedule(epoch):
    lr = keras.backend.get_value(model.optimizer.lr)
    if epoch % 1 == 0 and epoch != 0:
        lr = lr * 0.5
    return lr


log = keras.callbacks.TensorBoard(log_dir=logdir, update_freq=500)
reduceLr = keras.callbacks.LearningRateScheduler(lrSchedule, verbose=1)

model.fit_generator(generator=trainGenerator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epoch,
                    validation_data=evalGenerator,
                    validation_steps=10,
                    workers=1,
                    callbacks=[log, reduceLr])
model.save_weights(logdir + 'train_weight.h5')
