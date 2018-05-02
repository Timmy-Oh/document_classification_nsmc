# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np

# from sklearn.model_selection import KFold
from sklearn.utils import shuffle

### keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, add, multiply, average
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D, Dropout
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.layers import K, Activation
from keras.engine import Layer
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.utils import to_categorical


### from code
# from attention_layer import Attention
from models import models
from dataset import MovieReviewDataset, preprocess

def save(filename, *args):
    model.save_weights(filename)

def load(filename, *args):
    model.load_weights(filename)
    print('Model loaded')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # run config
    args.add_argument('--mode', type=str, default='train')
    
    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--strmaxlen', type=int, default=240)
    
    #rnn config
    # args.add_argument('--maxlen', type=int, default=200)
    args.add_argument('--cell_size_l1', type=int, default=50)
    args.add_argument('--cell_size_l2', type=int, default=40)
    args.add_argument('--embed_size', type=int, default=256)
    args.add_argument('--prob_dropout', type=float, default=0.3)
    args.add_argument('--prob_dropout2', type=float, default=0.2)

    #cnn config
    args.add_argument('--filter_size', type=int, default=32)
    args.add_argument('--kernel_size', type=int, default=3)
    args.add_argument('--strides', type=int, default=2)

    args.add_argument('--max_features', type=int, default=251)


    args.add_argument('--batch_size', type=int, default=1024)
    
    config = args.parse_args()

    DATA_PATH = '../data/nsmc-master/'
    DATASET_PATH = os.path.join(DATA_PATH, 'ratings_train.txt')
    TESTSET_PATH = os.path.join(DATA_PATH, 'ratings_test.txt')
        
    #model
    print("model creating...")
    model_builder = models() 
    model = model_builder.get_model1(config)
    model.summary()
    
    # Train Mode
    if config.mode == 'train':

        print("data loading...")
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        x_trn = np.array(dataset.reviews)
        y_trn = np.array(dataset.labels)
        
        x_tr, y_tr = shuffle(x_trn, y_trn, random_state=1991)
        
        # callbacks
        checkpoint = ModelCheckpoint('./best.hdf5', monitor='val_loss', save_best_only=True, mode='min', period=1)
        
        # testset
        testset = MovieReviewDataset(TESTSET_PATH, config.strmaxlen)
        x_te = np.array(testset.reviews)
        y_te = np.array(testset.labels)

        # label categorical
        y = to_categorical(y_tr, num_classes=2)
        y_val = to_categorical(y_te, num_classes=2)
        print("model training...")
        hist = model.fit(x_tr, y, 
                         validation_data = (x_te, y_val),
                         batch_size=config.batch_size, callbacks=[], epochs=config.epochs, verbose=2)

    # Test Mode
    elif config.mode == 'test':
        pass
        # with open(os.path.join(DATASET_PATH, 'test/test_data'), 'rt', encoding='utf-8') as f:
        #     reviews = f.readlines()
        # res = nsml.infer(reviews)
        # print(res)