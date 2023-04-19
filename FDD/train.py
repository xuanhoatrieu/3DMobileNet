from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
import keras_metrics as km
from sklearn.model_selection import train_test_split
from data_generator import *
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from MobileNet3D_2stream import MobileNet3D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(batch_size=16, NUM_FRAMES_PER_CLIP=16, IMAGE_SHAPE=224, nb_epoch=100):
    full_data = read_txt_files("Coffee_room") + read_txt_files("Home")
    new_data = video_split(full_data, NUM_FRAMES_PER_CLIP)
    random.shuffle(new_data)

    data_train, data_test = train_test_split(new_data, test_size=0.2, random_state=2019)
    print(count(data_train), len(data_train))
    print(count(data_test), len(data_test))

    steps_per_epoch = len(data_train) // batch_size
    val_steps_per_epoch = len(data_test) // batch_size

    model = MobileNet3D(input_shape=(NUM_FRAMES_PER_CLIP, IMAGE_SHAPE, IMAGE_SHAPE, 3),nb_classes=1)
    model.summary(line_length=130)
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy', km.binary_precision(), km.binary_recall()])
    early_stop = EarlyStopping(patience=20, verbose=2, monitor='val_loss', mode='min')
    checkpoint = ModelCheckpoint(filepath='MobileNet3D_2stream.hdf5', verbose=2, save_best_only=True, monitor='val_acc', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, verbose=2, monitor='val_loss', mode='min', min_lr=5e-7)
    call_back = [early_stop, checkpoint, reduce_lr]
    hist = model.fit_generator(generator=generator_data(data_train, batch_size),
                               steps_per_epoch=steps_per_epoch,
                               epochs=nb_epoch,
                               verbose=2,
                               validation_data=generator_data(data_test, batch_size),
                               validation_steps=val_steps_per_epoch,
                               callbacks=call_back, workers=4, use_multiprocessing=True)
    # Convert the history dictionary to a dataframe
    history_df = pd.DataFrame(hist.history)
    # Save the dataframe to a CSV file
    history_df.to_csv('FDD_3Dmobile_history.csv', index=False)
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
    plt.savefig('FDD_3Dmobile.png')
    # plt.show()

def main():
    NUM_FRAMES_PER_CLIP = 16
    IMAGE_SHAPE = 224
    BATCH_SIZE = 8
    NB_EPOCH = 100
    train(batch_size=BATCH_SIZE, NUM_FRAMES_PER_CLIP=NUM_FRAMES_PER_CLIP, IMAGE_SHAPE=IMAGE_SHAPE, nb_epoch=NB_EPOCH)

main()






