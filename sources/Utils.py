# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, LSTM, Dropout, Activation
from keras.preprocessing import sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, add, BatchNormalization, Conv1D, MaxPooling1D, Lambda

def get_train_data(maxlen, max_features, list_classes, tokenizer) :    
    train_file_name = "../input/train.csv"              
    train = pd.read_csv(train_file_name)    
    train = train.sample(frac=1, random_state=1)    
    list_sentences_train = train["comment_text"].fillna("CVxTz").values                 
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)    
    y = train[list_classes].values
    return X_train, y

def get_test_data(maxlen, max_features, tokenizer) :
    test_file_name = "../input/test.csv"
    test = pd.read_csv(test_file_name)   
    list_sentences_test = test["comment_text"].fillna("CVxTz").values        
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
    return X_test

def lambda_reverse(x):
    import tensorflow as tf
    return tf.reverse(x,axis=[0])

def conv_block(input_layer, conv_filters,dropout, act, prefix):
    """ Generic convolution block """
    inner = Conv1D(conv_filters, 3, padding='same',
                   activation=None, kernel_initializer='glorot_uniform',
                   name=prefix+"_conv")(input_layer)
    inner = BatchNormalization(name=prefix+"_bnorm")(inner)
    inner = MaxPooling1D(pool_size=2, 
                         name=prefix+"_maxp")(inner)
    inner = Dropout(dropout, name=prefix+"_drop")(inner)
    return Activation(act, name=prefix+"_act")(inner)

def rnn_block(input_layer, rnnLayer, rnn_size, merge, drop, rnn_drop, prefix):
    # normal rnn
    gru_1 = rnnLayer(rnn_size, return_sequences=True,
                     kernel_initializer='glorot_uniform',
                     recurrent_dropout=rnn_drop,
                     name=prefix+'_gruLR')(input_layer)
    #reversed rnn
    gru_1b = rnnLayer(rnn_size, return_sequences=True,
                      go_backwards=True,
                      kernel_initializer='glorot_uniform',
                      recurrent_dropout=rnn_drop,
                      name=prefix+'_gruRL')(input_layer)
    # lambda to reverse output
    gru_1b = Lambda(lambda_reverse)(gru_1b)    
    # merge
    if merge.lower() == "concat":
        gru1_merged = concatenate([gru_1, gru_1b])
    else:
        gru1_merged = add([gru_1, gru_1b])
        
    
    gru1_merged = BatchNormalization(name=prefix+'_bnorm')(gru1_merged)
    return Dropout(drop, name=prefix+'_drop')(gru1_merged)


def get_model(maxlen, max_features, num_classes=6, conv_filters=128, rnn_size = 128, drop = 1.0, rnn_drop = 0.0, rnn = "GRU"):             
    embed_size = 128
    input_layer = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(input_layer)
    """ Conv1D + Separable + RNN
    num_classes: number of output classes
    """   
    act = 'relu'

    rnnLayer = GRU
    if rnn == "LSTM":
        rnnLayer = LSTM
   
    inner = conv_block(x, conv_filters,drop*.50, act, "cblock1")        
    # Reduce size before rnn
    inner = Dense(64, activation=act, kernel_initializer='glorot_uniform', name='dense1')(inner)
    inner = Dropout(0.2,name='drop3')(inner)
    gru_size = int(rnn_size/2)
    # Temporal filters
    gru1 = rnn_block(inner, rnnLayer, gru_size, "add", drop*.50, rnn_drop, prefix = "rblock1")    
    inner = conv_block(gru1, int(conv_filters/2),drop*.75, act, "cblock2")    
    gru_size = rnn_size
    gru2 = rnn_block(inner, rnnLayer, gru_size, "concat", drop*.75, rnn_drop, prefix = "rblock2")        
    avg_pool = GlobalAveragePooling1D()(gru2)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    ## TEXT CLASSIFICATION
    text_pred = Dense(num_classes, activation="sigmoid", kernel_initializer='glorot_uniform', name = 'text_dense')(x)
    
    model = Model(inputs=input_layer, outputs=text_pred)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model