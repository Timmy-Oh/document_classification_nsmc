# keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, add, multiply, average
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, Conv1D, Dropout
from keras.layers import Dropout, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.layers import K, Activation
from keras.engine import Layer
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint

### from code
from attention_layer import Attention

class models:
    def get_baseline(self, config):
            inp = Input(shape=(config.strmaxlen, ), name='input')
            emb = Embedding(config.max_features, config.max_features,embeddings_initializer='identity', trainable = True)(inp)
            # emb1 = SpatialDropout1D(config.prob_dropout)(emb)
            
            #### 
            l1_L = Bidirectional(CuDNNGRU(config.cell_size_l1, return_sequences=True))(emb)
            
            l2_LL = Bidirectional(CuDNNGRU(config.cell_size_l2, return_sequences=False))(l1_L)
            
            # l3_LLC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_LL)

            # avg_pool_L = GlobalAveragePooling1D()(l1_L)
            # max_pool_L = GlobalMaxPooling1D()(l1_L)
            
            # avg_pool_LL = GlobalAveragePooling1D()(l2_LL)
            # max_pool_LL = GlobalMaxPooling1D()(l2_LL)
            
            # attention_LLA = Attention(config.strmaxlen)(l2_LL)

            # avg_pool_LLC = GlobalAveragePooling1D()(l3_LLC)
            # max_pool_LLC = GlobalMaxPooling1D()(l3_LLC)
            
            # attention_LLCA = Attention(int(config.strmaxlen/2-1))(l3_LLC)
            # attention_LGCA = Attention(int(config.strmaxlen/2-1))(l3_LGC)
            
            # conc_LLC = concatenate([avg_pool_L, max_pool_L, avg_pool_LL, max_pool_LL, avg_pool_LLC, max_pool_LLC, attention_LLA, attention_LLCA])
            # conc_LGC = concatenate([avg_pool_L, max_pool_L, avg_pool_LG, max_pool_LG, avg_pool_LGC, max_pool_LGC, attention_LGA, attention_LGCA])        

            # out_LL = Dropout(config.prob_dropout2)(max_pool_LL)
            out_LL = Dense(2, activation='softmax')(l2_LL)
            
    #       ==================================================================================================
            model = Model(inputs=inp, outputs=out_LL)
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=0.00), metrics=['categorical_crossentropy', 'accuracy'])
            
            return model

    def get_model1(self, config):
            inp = Input(shape=(config.strmaxlen, ), name='input')
            emb = Embedding(config.max_features, config.max_features,embeddings_initializer='identity', trainable = True)(inp)
            emb1 = SpatialDropout1D(config.prob_dropout)(emb)
            
            #### 
            l1_L = Bidirectional(CuDNNGRU(config.cell_size_l1, return_sequences=True))(emb)
            l2_LL = Bidirectional(CuDNNGRU(config.cell_size_l2, return_sequences=True))(l1_L)
            l3_LLC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_LL)
            attention_LLA = Attention(config.strmaxlen)(l2_LL)

            avg_pool_L = GlobalAveragePooling1D()(l1_L)
            max_pool_L = GlobalMaxPooling1D()(l1_L)
            avg_pool_LL = GlobalAveragePooling1D()(l2_LL)
            max_pool_LL = GlobalMaxPooling1D()(l2_LL)
            avg_pool_LLC = GlobalAveragePooling1D()(l3_LLC)
            max_pool_LLC = GlobalMaxPooling1D()(l3_LLC)
            attention_LLCA = Attention(int(config.strmaxlen/2-1))(l3_LLC)
            
            conc_LLC = concatenate([avg_pool_L, max_pool_L, avg_pool_LL, max_pool_LL, avg_pool_LLC, max_pool_LLC, attention_LLA, attention_LLCA])
            out_LL = Dropout(config.prob_dropout2)(conc_LLC)
            out_LL = Dense(2, activation='softmax')(out_LL)
            
    #       ==================================================================================================
            model = Model(inputs=inp, outputs=out_LL)
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=0.00), metrics=['categorical_crossentropy', 'accuracy'])
            
            return model

    def get_model(self, config):
            inp = Input(shape=(config.strmaxlen, ), name='input')
    #         inp = Input(shape=(config.max_features, ), name='input')

            emb = Embedding(config.max_features, config.max_features,embeddings_initializer='identity', trainable = True)(inp)
    #         emb1 = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
            emb1 = SpatialDropout1D(config.prob_dropout)(emb)
            
            #### 
            l1_L = Bidirectional(CuDNNLSTM(config.cell_size_l1, return_sequences=True))(emb1)
            
            l2_LL = Bidirectional(CuDNNLSTM(config.cell_size_l2, return_sequences=True))(l1_L)
            l2_LG = Bidirectional(CuDNNGRU(config.cell_size_l2, return_sequences=True))(l1_L)
            
            l3_LLC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_LL)
            l3_LGC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_LG)

            avg_pool_L = GlobalAveragePooling1D()(l1_L)
            max_pool_L = GlobalMaxPooling1D()(l1_L)
            
            
            avg_pool_LL = GlobalAveragePooling1D()(l2_LL)
            max_pool_LL = GlobalMaxPooling1D()(l2_LL)
            avg_pool_LG = GlobalAveragePooling1D()(l2_LG)
            max_pool_LG = GlobalMaxPooling1D()(l2_LG)
            
            attention_LLA = Attention(config.strmaxlen)(l2_LL)
            attention_LGA = Attention(config.strmaxlen)(l2_LG)

            avg_pool_LLC = GlobalAveragePooling1D()(l3_LLC)
            max_pool_LLC = GlobalMaxPooling1D()(l3_LLC)
            avg_pool_LGC = GlobalAveragePooling1D()(l3_LGC)
            max_pool_LGC = GlobalMaxPooling1D()(l3_LGC)
            
            attention_LLCA = Attention(int(config.strmaxlen/2-1))(l3_LLC)
            attention_LGCA = Attention(int(config.strmaxlen/2-1))(l3_LGC)
            
            conc_LLC = concatenate([avg_pool_L, max_pool_L, avg_pool_LL, max_pool_LL, avg_pool_LLC, max_pool_LLC, attention_LLA, attention_LLCA])
            conc_LGC = concatenate([avg_pool_L, max_pool_L, avg_pool_LG, max_pool_LG, avg_pool_LGC, max_pool_LGC, attention_LGA, attention_LGCA])        

            out_LL = Dropout(config.prob_dropout2)(conc_LLC)
            out_LG = Dropout(config.prob_dropout2)(conc_LGC)
            out_LL = Dense(2, activation='softmax')(out_LL)
            out_LG = Dense(2)(out_LG)
            ####
            
    #         emb2 = Embedding(config.max_features, config.max_features,embeddings_initializer='identity', trainable = True)(inp)
    #         emb1 = Embedding(config.max_features, config.embed_size, trainable = True)(inp)
            emb2 = SpatialDropout1D(config.prob_dropout)(emb)
            
            #### 
            l1_G = Bidirectional(CuDNNGRU(config.cell_size_l1, return_sequences=True))(emb2)
            
            l2_GL = Bidirectional(CuDNNLSTM(config.cell_size_l2, return_sequences=True))(l1_G)
            l2_GG = Bidirectional(CuDNNGRU(config.cell_size_l2, return_sequences=True))(l1_G)
            
            l3_GLC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_GL)
            l3_GGC = Conv1D(config.filter_size, kernel_size = config.kernel_size, strides=2, padding = "valid", kernel_initializer = "he_uniform")(l2_GG)

            avg_pool_G = GlobalAveragePooling1D()(l1_G)
            max_pool_G = GlobalMaxPooling1D()(l1_G)
            
            
            avg_pool_GL = GlobalAveragePooling1D()(l2_GL)
            max_pool_GL = GlobalMaxPooling1D()(l2_GL)
            avg_pool_GG = GlobalAveragePooling1D()(l2_GG)
            max_pool_GG = GlobalMaxPooling1D()(l2_GG)
            
            attention_GLA = Attention(config.strmaxlen)(l2_GL)
            attention_GGA = Attention(config.strmaxlen)(l2_GG)

            avg_pool_GLC = GlobalAveragePooling1D()(l3_GLC)
            max_pool_GLC = GlobalMaxPooling1D()(l3_GLC)
            avg_pool_GGC = GlobalAveragePooling1D()(l3_GGC)
            max_pool_GGC = GlobalMaxPooling1D()(l3_GGC)
            
            attention_GLCA = Attention(int(config.strmaxlen/2-1))(l3_GLC)
            attention_GGCA = Attention(int(config.strmaxlen/2-1))(l3_GGC)
            
            conc_GLC = concatenate([avg_pool_G, max_pool_G, avg_pool_GL, max_pool_GL, avg_pool_GLC, max_pool_GLC, attention_GLA, attention_GLCA])
            conc_GGC = concatenate([avg_pool_G, max_pool_G, avg_pool_GG, max_pool_GG, avg_pool_GGC, max_pool_GGC, attention_GGA, attention_GGCA])        

            out_GL = Dropout(config.prob_dropout2)(conc_GLC)
            out_GG = Dropout(config.prob_dropout2)(conc_GGC)
            out_GL = Dense(1)(out_GL)
            out_GG = Dense(1)(out_GG)
            
            out_avg = average([out_LL, out_LG, out_GL, out_GG])

            
    # #         ==================================================================================================
            model_avg = Model(inputs=inp, outputs=[out_LL, out_LG, out_GL, out_GG, out_avg])
            
    #         inp_pre = Input(shape=(config.strmaxlen, ), name='input_pre')
    #         inp_post = Input(shape=(config.strmaxlen, ), name='input_post')
            
    #         model_pre = model_avg(inp_pre)
    #         model_post = model_avg(inp_post)
            
    #         stack_layer = concatenate([model_pre, model_post])
    #         ens_out = Dense(1, use_bias=False)(stack_layer)
            
    #         reg_model = Model(inputs=[inp_pre, inp_post], outputs=ens_out)
            
            model_avg.compile(loss='mean_squared_error', optimizer='adam',
                            loss_weights=[1., 1., 1., 1., 0.1],
                            metrics=['mean_squared_error', 'accuracy'])
            
            return model_avg