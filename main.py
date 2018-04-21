import servizi 
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, print_summary
from keras.preprocessing import sequence
import keras
import json
import Model as m
import copy
import Model_pos as mp
import random


max_words=2000
np.random.seed(42)
dictionary_path ='dizionario_for_keras_raw.json'
dictionary_path_token_pos = 'dizionario_token_pos.json'
dictionary_path_pos = 'dizionario_pos.json'
dictionary_path_labels = 'dizionario_labels.json'
embedd='glove'
embedding_path = 'glove.6B.200d.txt'
#attenzione se si usa il glove bisogna settare a 200(dipende dal tipo di glove), se si usa gensim bisogna settare a 300
batch_size = 128
EMBED_SIZE = 200

path = 'Sentiment-Train.xls'
#leggo dal file xls le varie sentences e le corrispondenti emozioni, facendole restituire in due liste
sentences,labels = servizi.readFile(path)
#raggruppo le sentences nelle corrispondenti emozioni e le scrivo su un dizionario dove la key è l'emozione e il valore è un alista delle sentences
dizionario_sentences = servizi.divisioneSentences(sentences,labels)
#Mi faccio restituire la lunghezza più piccola delle varie liste di sentences
valore_minimo = servizi.minimo(dizionario_sentences)
#mi faccio restituire un dizionario dove la lista delle sentences è compresa un intervallo [start,stop] 
new_dizionario = servizi.frasiBilanciate(dizionario_sentences,0,valore_minimo)
#set = contiene 20 sentences di ciascuna emozione
#emot = contiene le corrispondenti emozioni delle sentences
#lista_tot = contiene le restanti sentences (tot_sent - 20 * (num_emotion))
sent,emot,lista_tot = servizi.primaDivisione(new_dizionario,20)
#faccio il clustering su lista_tot e mi faccio restituire un dizionario
clust = servizi.get_clusters('clustering.json',lista_tot)


#PREPARAZIONE PER L'ADDESTRAMENTO

i = 1
while len(lista_tot) > len(sent):

    #---------------------- RAW TEXT -------------------#
    
    print('Iterazione numero: ',i)
    print('MODELLO RAW TEXT')
    dictionary,train_x = servizi.createDictionary(sent,dictionary_path,max_words)
    labels,l_ed = servizi.convert(emot)
    dicionary_size = len(dictionary)
    max_sentence_length = servizi.maxLenSentences(train_x)
    embedding_weights,EMBED_SIZE = servizi.load_word_embedding(embedd,embedding_path,dictionary,EMBED_SIZE)
    num_emotion = max(labels)
    #random_state di default è None. Indica il seme utilizzato dal generatore di numeri casuali
    #shuffle di default è true e indica se mescolare i dati prima della divisione
    train_x,test_x,train_labels,test_labels = train_test_split(train_x,labels,test_size=0.30,random_state = 42,shuffle=False)
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)
	
    print('Pad sequences (samples x time)')
    #padding inserire gli 0 o prima o dopo la lunghezza della frase
    #value = 0 default
    train_x = sequence.pad_sequences(train_x, maxlen=max_sentence_length,padding='pre')
    test_x = sequence.pad_sequences(test_x, maxlen=max_sentence_length)
    print('x_train shape:', train_x.shape)
    print('x_test shape:', test_x.shape)
    
	#Inizializzazione del modello
    model_raw = m.Model(embedding_weights,dicionary_size,max_sentence_length,num_emotion)
    model_raw.model.fit(train_x, train_labels,batch_size=batch_size,epochs=10,validation_data=[test_x, test_labels])
    loss, accuracy = model_raw.model.evaluate(train_x, train_labels, verbose=1)
    print('Accuracy train: %f' % (accuracy*100))
    print('Loss train: %f' % (loss*100))

    loss_test, accuracy_test = model_raw.model.evaluate(test_x, test_labels, verbose=1)
    print('Accuracy test: %f' % (accuracy_test*100))
    print('Loss  test: %f' % (loss_test*100))
	
	#predico le frasi totali
    date = []
    for s in lista_tot:
        date.append(servizi.prepareData(dictionary,s))
    date = sequence.pad_sequences(maxlen=max_sentence_length,sequences=date,padding="post",value=0)
    pred = model_raw.model.predict(date)
    servizi.etichettatura(pred,lista_tot,l_ed,'raw_test.xls')
    
	#--------------------POS----------------------#
    print('MODELLO POS')
    pos = servizi.IOB(sent)
    lista_pos,lista_words = servizi.list_unique(pos)
    dizionario_word,dizionario_pos= servizi.createDictionaryTokenPos(dictionary_path_token_pos,dictionary_path_pos,lista_words,lista_pos)
    token_pos,labels_int,labels_unique_text = servizi.convertDate(pos,dizionario_word,dizionario_pos,emot)
    embedding_weights_pos,EMBED_SIZE_POS = servizi.load_word_embedding(embedd,embedding_path,dizionario_word,EMBED_SIZE)
    max_sentence_length_pos = servizi.maxLenSentences(token_pos)
    #random_state di default è None. Indica il seme utilizzato dal generatore di numeri casuali
    #shuffle di default è true e indica se mescolare i dati prima della divisione
    train_x_pos,test_x_pos,train_labels_pos,test_labels_pos = train_test_split(token_pos,labels_int,test_size=0.30,random_state = 42,shuffle=False)
    train_labels_pos = keras.utils.to_categorical(train_labels_pos)
    test_labels_pos = keras.utils.to_categorical(test_labels_pos)
	
    print('Pad sequences (samples x time)')
    #padding inserire gli 0 o prima o dopo la lunghezza della frase
    #value = 0 default
    train_x_pos = sequence.pad_sequences(train_x_pos, maxlen=max_sentence_length_pos,padding='pre')
    test_x_pos = sequence.pad_sequences(test_x_pos, maxlen=max_sentence_length_pos)
    print('x_train shape:', train_x_pos.shape)
    print('x_test shape:', test_x_pos.shape)
	
    model_pos = mp.Model(embedding_weights_pos,len(dizionario_word),max_sentence_length_pos,num_emotion)
    model_pos.model.fit(train_x_pos, train_labels_pos,batch_size=batch_size,epochs=10,validation_data=[test_x_pos, test_labels_pos])
	
    loss_pos, accuracy_pos = model_pos.model.evaluate(train_x_pos, train_labels_pos, verbose=1)
    print('Accuracy train: %f' % (accuracy_pos*100))
    print('Loss train: %f' % (loss_pos*100))

    loss_test_pos, accuracy_test_pos = model_pos.model.evaluate(test_x_pos, test_labels_pos, verbose=1)
    print('Accuracy test: %f' % (accuracy_test_pos*100))
    print('Loss  test: %f' % (loss_test_pos*100))
    date_pos = servizi.prepareDatePos(dizionario_word,dizionario_pos,lista_tot)
    date_pos = sequence.pad_sequences(maxlen=max_sentence_length_pos,sequences=date_pos,padding="post",value=0)
    pred_pos = model_pos.model.predict(date_pos)
    servizi.etichettatura(pred_pos,lista_tot,labels_unique_text,'pos_test.xls')
	
    sent_comuni,lab_comuni = servizi.confronto('raw_test.xls','pos_test.xls')
    k = servizi.numMinimoFrasiCluster(clust,sent_comuni)
    s,em = servizi.deleteSentCluster(clust,sent_comuni,lab_comuni,k)
    sent += s
    emot += em
    random.Random(5).shuffle(sent)
    random.Random(5).shuffle(emot)
    lista_tot = servizi.deleteSentTot(lista_tot,s)
    i += 1
    print(len(lista_tot))
    print(len(sent))
    print(len(emot))	

	#Ultima istruzione togliere le frasi migliori da frasi tot

