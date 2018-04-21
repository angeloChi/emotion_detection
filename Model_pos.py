from keras.layers import  Embedding, Dense, TimeDistributed, Dropout, Conv1D,GlobalMaxPooling1D,Flatten,Input
import keras
from keras.models import Model as KerasModel
from keras.preprocessing.text import Tokenizer

class Model(object):
    EMBEDDING_WORD_DIM = 200
    def themodel (embedding_weights, dictionary_size, MAX_SEQUENCE_LENGHT,emotion):
        input_seq = Input(shape=(MAX_SEQUENCE_LENGHT,2,))
        embed = TimeDistributed(Embedding(input_dim=dictionary_size+1,weights=[embedding_weights],output_dim=Model.EMBEDDING_WORD_DIM,input_length=None,mask_zero=True,name="embed"))(input_seq)
        x = Dropout(0.2)(embed)
        x = TimeDistributed(Flatten())(x)
        x = Conv1D(filters=100,kernel_size=3,activation="relu")(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.2)(x)
        preds = Dense(emotion+1,activation="softmax")(x)
        model = KerasModel(input_seq,preds)
        return model
	
    def __init__(self,embedding_weights,dictionary_size,MAX_SEQUENCE_LENGHT,emotion):
        self.model = Model.themodel(embedding_weights,dictionary_size,MAX_SEQUENCE_LENGHT,emotion)
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])