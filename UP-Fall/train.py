from config import get_parser
import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD
from data_new import get_data, generator_data
from model_tensor2 import MobileNet3D
import keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# K.set_session(session)

def train(cfg):
    # Prepare data for training phase
    input_shape = (cfg.n_frames, cfg.image_size, cfg.image_size, cfg.n_channel)
    train_dataset = get_data('train.csv')
    test_dataset = get_data('test.csv')
    steps_per_epoch = len(train_dataset) * 10 // cfg.batch_size
    val_steps_per_epoch = len(test_dataset) // cfg.batch_size

    # Build model
    model = MobileNet3D(input_shape=input_shape, nb_classes=1, pooling='avg', alpha=0.25, depth_multiplier=1, activation='sigmoid')
    model.summary(line_length=150)

    # Build callbacks
    optimizer = SGD(learning_rate=cfg.lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(patience=20, verbose=2, monitor='val_loss', mode='min')
    checkpoint = ModelCheckpoint(filepath='MobileNet3D_best.hdf5', verbose=2, save_best_only=True, save_weights_only=True,
                                 monitor='val_acc', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, verbose=2, monitor='val_loss', mode='min', min_lr=5e-7)
    call_back = [early_stop, checkpoint, reduce_lr]

    # Training
    hist = model.fit_generator(generator=generator_data(train_dataset, cfg.batch_size, cfg.n_frames, cfg.image_size, is_train=True),
                               steps_per_epoch=steps_per_epoch,
                               epochs=cfg.epochs,
                               verbose=2,
                               validation_data=generator_data(test_dataset, cfg.batch_size, cfg.n_frames, cfg.image_size, is_train=False),
                               validation_steps=val_steps_per_epoch,
                               callbacks=call_back, workers=4, use_multiprocessing=True)

def main(cfg):
    train(cfg)

if __name__ == '__main__':
    cfg = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)  # Choose GPU for training
    main(cfg)

