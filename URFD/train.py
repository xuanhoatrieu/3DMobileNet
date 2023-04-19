from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from keras import backend as K
import keras_metrics as km
# from tensorflow.keras.metrics import BinaryAccuracy, SensitivityAtSpecificity, SpecificityAtSensitivity
from sklearn.model_selection import train_test_split
from gen import *
import time
import os
import pandas as pd
import numpy  as np
import h5py
import matplotlib.pyplot as plt
from model import MobileNet3D
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def train(batch_size=16, NUM_FRAMES_PER_CLIP=16, IMAGE_SHAPE=224, nb_epoch=100):
    new_data = read_csvfile('full.csv', NUM_FRAMES_PER_CLIP)
    # random.shuffle(new_data)

    # data_train, data_test = train_test_split(new_data, test_size=0.2, random_state=2019)

    data_train = get_data('train_3dmobile.csv')
    data_test = get_data('test_3dmobile.csv')
    print(count(data_train), len(data_train))
    print(count(data_test), len(data_test))

    steps_per_epoch = len(data_train) // batch_size
    val_steps_per_epoch = len(data_test) //batch_size

    model = MobileNet3D(input_shape=(NUM_FRAMES_PER_CLIP, IMAGE_SHAPE, IMAGE_SHAPE, 3),nb_classes=1)
    # model = ShuffleNet_3D(input_shape=(NUM_FRAMES_PER_CLIP, IMAGE_SHAPE, IMAGE_SHAPE, 3),nb_classes=1)
    model.summary(line_length=130)
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', km.binary_precision(), km.binary_recall()])
    early_stop = EarlyStopping(patience=20, verbose=1, monitor='val_loss', mode='min')
    checkpoint = ModelCheckpoint(filepath='3dmobile_urfd_best.h5', verbose=2, save_best_only=True, monitor='val_acc', mode='max', save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, verbose=2, monitor='val_loss', mode='min', min_lr=5e-7)
    call_back = [early_stop, checkpoint, reduce_lr]
    hist = model.fit_generator(generator=generator_data(data_train*10, batch_size),
                               steps_per_epoch=steps_per_epoch,
                               epochs=nb_epoch,
                               verbose=1,
                               validation_data=test_data(data_test*10, batch_size),
                               validation_steps=val_steps_per_epoch,
                               callbacks=call_back, workers=12, use_multiprocessing=True)
    # use_multiprocessing=True                        
    # Convert the history dictionary to a dataframe
    history_df = pd.DataFrame(hist.history)
    # Save the dataframe to a CSV file
    history_df.to_csv('UR_3Dmobile_history.csv', index=False)
    pd.DataFrame(hist.history).to_hdf('mobilenet3d.h5', "history")
    plt.subplot(211)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.title('Loss')
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('UR_3Dmobile.png')
    # plt.show()

def testing(batch_size=16, NUM_FRAMES_PER_CLIP=16, IMAGE_SHAPE=224, nb_epoch=100):
    data_test = get_data('test_3dmobile.csv')

    model = MobileNet3D(input_shape=(NUM_FRAMES_PER_CLIP, IMAGE_SHAPE, IMAGE_SHAPE, 3),nb_classes=1)
    # model = ShuffleNet_3D(input_shape=(NUM_FRAMES_PER_CLIP, IMAGE_SHAPE, IMAGE_SHAPE, 3),nb_classes=1)
    model.summary(line_length=130)
    model.load_weights('3dmobile_urfd_best.h5')
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', km.binary_precision(), km.binary_recall()])
    
    _, accuracy, precision, recall = model.evaluate(data_test, verbose=1, steps=(len(data_test) // batch_size))
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    print('---------------Results on the test dataset--------------')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recal: ', recall)
    print('F1 score: ', f1_score)

def predic(batch_size=16, NUM_FRAMES_PER_CLIP=16, IMAGE_SHAPE = 224):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_test = get_data('test_3dmobile.csv')
    y_test=[]
    for row in data_test:
        y_test.append(int(row[-1]))    
    
    ds_test = test_data(data_test, 1)
    # ds_test = ds_test.batch(1).prefetch(AUTOTUNE)
    model = MobileNet3D(input_shape=(NUM_FRAMES_PER_CLIP, IMAGE_SHAPE, IMAGE_SHAPE, 3),nb_classes=1)
    # model = ShuffleNet_3D(input_shape=(NUM_FRAMES_PER_CLIP, IMAGE_SHAPE, IMAGE_SHAPE, 3),nb_classes=1)
    # model.summary(line_length=130)
    file_path = '3dmobile_urfd_best.h5'
    model.load_weights('3dmobile_urfd_best.h5')

    print(model.get_weights())
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', km.binary_precision(), km.binary_recall()])
    
    y_predict = model.predict(ds_test, steps=len(y_test))
    print(len(y_predict))

    y_predict[y_predict>0.5] = 1
    y_predict[y_predict<=0.5] = 0
    np.savez('3dmobile_ur_y_predict.npz', data=y_predict)

    # x = np.load('y_predict.npz')['data']
    y_test = np.asarray(y_test)
    cm = confusion_matrix(y_test, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('3dmobile_ur_confusion_matrix.png')
    plt.show()

def main():
    NUM_FRAMES_PER_CLIP = 16
    IMAGE_SHAPE = 224
    BATCH_SIZE = 16
    NB_EPOCH = 100

    # train(batch_size=BATCH_SIZE, NUM_FRAMES_PER_CLIP=NUM_FRAMES_PER_CLIP, IMAGE_SHAPE=IMAGE_SHAPE, nb_epoch=NB_EPOCH)
    
    # testing(batch_size=BATCH_SIZE, NUM_FRAMES_PER_CLIP=NUM_FRAMES_PER_CLIP, IMAGE_SHAPE=IMAGE_SHAPE, nb_epoch=NB_EPOCH)
    predic(batch_size=BATCH_SIZE, NUM_FRAMES_PER_CLIP=NUM_FRAMES_PER_CLIP, IMAGE_SHAPE = IMAGE_SHAPE)

main()

