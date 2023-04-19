import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

import argparse
import random
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy, MSE
from tensorflow.keras.metrics import BinaryAccuracy, SensitivityAtSpecificity, SpecificityAtSensitivity, Recall, Precision
# from tensorflow.keras.metrics import Accuracy, Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import get_parser
from data_new import get_data, generator_training_data, data_augmentation, generator_test_data
from model_tensor2 import MobileNet3D
from epochcheckpoint import EpochCheckpoint

def training(train_dataset, test_dataset, args, save_path='save_model'):
    input_shape = (args.n_frames, args.image_size, args.image_size, args.n_channel)
    # Prepare data for training phase
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_train = tf.data.Dataset.from_generator(generator_training_data,
                                              (tf.float32, tf.float32),
                                              (tf.TensorShape(input_shape), tf.TensorShape([])),
                                              args=[train_dataset, input_shape[0], input_shape[1]])
    ds_train = ds_train.map(data_augmentation, num_parallel_calls=AUTOTUNE).batch(args.batch_size).prefetch(AUTOTUNE)

    ds_test = tf.data.Dataset.from_generator(generator_test_data, (tf.float32, tf.float32),
                                             (tf.TensorShape(input_shape), tf.TensorShape([])),
                                             args=[test_dataset, input_shape[0], input_shape[1]])
    ds_test = ds_test.batch(args.batch_size).prefetch(AUTOTUNE)

    model = MobileNet3D(input_shape=input_shape)
    model.summary(line_length=150)

    # Build callbacks
    optimizer = SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    # metrics = [Accuracy(), Recall(), Precision()]
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    early_stop = EarlyStopping(patience=20, verbose=2, monitor='val_loss', mode='min')
    checkpoint = ModelCheckpoint(filepath='MobileNet3D_best.h5', verbose=2, save_best_only=True,
                                 save_weights_only=True, monitor='val_accuracy', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, verbose=2, monitor='val_loss', mode='min', min_lr=5e-7)
    outputPath = os.path.join(save_path, "checkpoints")
    epoch_checkpoint = EpochCheckpoint(outputPath=outputPath, every=1, startAt=args.start_epoch)
    call_back = [early_stop, checkpoint, reduce_lr, epoch_checkpoint]
    # Training
    history = model.fit(ds_train, epochs=args.epochs, verbose=1,
                 steps_per_epoch=(len(train_dataset) * 10 // args.batch_size),
                #  steps_per_epoch=10,
                 validation_data=ds_test,
                 validation_steps=(len(test_dataset)*10 // args.batch_size),
                 #validation_steps=10,
                 callbacks=call_back
                 )
    # Convert the history dictionary to a dataframe
    history_df = pd.DataFrame(history.history)
    # Save the dataframe to a CSV file
    history_df.to_csv('history.csv', index=False)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')             
    # plt.savefig("3DMobileNet_model.jpg")

def testing(test_dataset, args, save_path='save_model'):
    input_shape = (args.n_frames, args.image_size, args.image_size, args.n_channel)
    # Prepare data for training phase
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_test = tf.data.Dataset.from_generator(generator_test_data, (tf.float32, tf.float32),
                                             (tf.TensorShape(input_shape), tf.TensorShape([])),
                                             args=[test_dataset, input_shape[0], input_shape[1]])
    ds_test = ds_test.batch(args.batch_size).prefetch(AUTOTUNE)

    model = MobileNet3D(input_shape=input_shape)
    model.summary(line_length=150)
    model.load_weights('MobileNet3D_best.h5')

    # Build callbacks
    optimizer = SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    # metrics = [Accuracy(), Recall(), Precision()]
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    # testing
    loss, BinaryAccuracy, Recall, Precision = model.evaluate(ds_test, verbose=1,
                 steps=(len(test_dataset)*10 // args.batch_size)
                 )
    # Compute and print F1 score
    f1_score = 2 * (Precision * Recall) / (Precision + Recall + 1e-7)
    # print('Test accuracy:', accuracy)
    # print('Test recall:', recall)
    # print('Test precision:', precision)
    # print('Test F1 score:', f1_score)

def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Choose GPU for training

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    train_dataset = get_data('train.csv')
    test_dataset = get_data('test.csv')
    random.shuffle(test_dataset)
    random.shuffle(train_dataset)
    print('Train set:', len(train_dataset))
    print('Test set:', len(test_dataset))

    save_path = 'save_model'

    # Create folders for callback
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, "output")):
        os.mkdir(os.path.join(save_path, "output"))
    if not os.path.exists(os.path.join(save_path, "checkpoints")):
        os.mkdir(os.path.join(save_path, "checkpoints"))

    # --------------------------------------Training ----------------------------------------
    training(train_dataset, test_dataset, args, save_path=save_path)

    # --------------------------------------Testing ----------------------------------------
    testing(test_dataset, args, save_path=save_path)




if __name__ == '__main__':
    print(tf.__version__)
    main()