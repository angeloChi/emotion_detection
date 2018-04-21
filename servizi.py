from pandas import DataFrame, read_csv
import json
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text as kpt
import pandas as pd
import numpy as np
from pathlib import Path
from keras.preprocessing.sequence import _remove_long_seq
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.models import Model
import itertools
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
import nltk
import copy

#Metodo che legge il file xls e restituisce le sentence e le corrispetive emozioni
def readFile(path):
    df = pd.read_excel(path)
	#ho una matrice di tipo numpy che contiene tutte le mie informazioni
    matrix = df.as_matrix()
	#assegno ad un array di tipo numpy le frasi con caus uguale a 2 o 3
    array_filtrato_sentence = np.asarray([x[40] for x in matrix if x[29]==2 or x[29]==3])
    #array_filtrato_sentence = np.asarray([x[1] for x in matrix])
	#assegno a un array di tipo numpy le emozioni con caus uguale a 2 o 3
    array_filtrato_emotion = np.asarray([x[36] for x in matrix if x[29]==2 or x[29]==3])
    #array_filtrato_emotion = np.asarray([x[0] for x in matrix])
    return array_filtrato_sentence,array_filtrato_emotion
	
#restituisce una lista che contiene le emozioni uniche
def uniqueEmotion(labels):
    return list(set(labels))

#Crea un dizionario dove la key è l'emozione e il valore è una lista che contiene le sentences riferita a quella emozione e salva tutto su un file json
def divisioneSentences(sentences,labels):
    diz_sent = dict()
    labels_unique = uniqueEmotion(labels)
    for emotion in labels_unique:
        x = str(emotion)
        diz_sent.setdefault(x,list())
    for i in range(0,len(sentences)):
	    (diz_sent.get(labels[i])).append(sentences[i])	
    return diz_sent

#restituisce la lunghezza più piccole delle varie liste
def minimo(dizionario_sent_emotion):
    y = dizionario_sent_emotion.values()
    lista_len_value = list()
    for k in y:
        lista_len_value.append(len(k))
    return min(lista_len_value)		

#metodo che prende in input il dizionario e un intervallo dando in output un nuovo dizionario con lo stesso numero di esempi per emozione(andiamo a verificare l'insieme più piccolo in questo caso guilt)
def frasiBilanciate(dizionario_sent_emotion,start,stop):
    new_dizionario = dict()
    for key in dizionario_sent_emotion.keys():
        lista = dizionario_sent_emotion.get(key)
        new_list = lista[start:stop]
        new_dizionario.setdefault(str(key),new_list)
    return new_dizionario

#metodo che viene invocato prima di effettuare l'iterazione tra i due modelli, esso mi permette di selezionare 140 
#frasi bilanciate per il training e di sottrarle al dataset totale in questo caso avremo 140 frasi etichettate e 1442-140=1302 frasi non etichettate 
#per effettuare co-training  	
def primaDivisione(dizionario_sent_emotion,num_example):
    lista_sent = list()
    lista_emotion = list()
    lista_tot_sent = list()
    for key,value in dizionario_sent_emotion.items():
        lista_emotion.append([key for i in range(0,num_example)])
        lista_sent.append(value[0:num_example])
        lista_tot_sent.append(value)
    mylist_sent = list(itertools.chain.from_iterable(lista_sent))
    mylist_emotion = list(itertools.chain.from_iterable(lista_emotion))
    mylist_tot = list(itertools.chain.from_iterable(lista_tot_sent))
    seed = 3
    random.Random(seed).shuffle(mylist_sent)
    random.Random(seed).shuffle(mylist_emotion)
    for sent in mylist_sent:
        mylist_tot.remove(sent)
    return mylist_sent,mylist_emotion,mylist_tot
    
 


#CLUSTERING
#Metodi che permettono di effettuare il clustering sulle mie frasi rimanenti(da utilizzare per il co-training),
#tali frasi vengono raggruppate in cluster per affinità

def stem_tokens(tokens):
    stemmer = nltk.stem.snowball.EnglishStemmer()
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    punctuation_map = dict((ord(char), None) for char in string.punctuation)
    return stem_tokens(nltk.word_tokenize(text.lower().translate(punctuation_map)))

def get_clusters(path,sentences):
    vectorizer = TfidfVectorizer(tokenizer=normalize)
    tf_idf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = (tf_idf_matrix * tf_idf_matrix.T).A
    affinity_propagation = AffinityPropagation(affinity="precomputed", damping=0.5)
    affinity_propagation.fit(similarity_matrix)

    labels = affinity_propagation.labels_
    cluster_centers = affinity_propagation.cluster_centers_indices_

    tagged_sentences = zip(sentences, labels)
    clusters = {}
    
    for sentence, cluster_id in tagged_sentences:
        clusters.setdefault(sentences[cluster_centers[cluster_id]], []).append(sentence)
    with open(path,'w') as clust:
        json.dump(clusters, clust)
    return clusters

	#FINE METODI CLUSTERING

#Metodo che crea il dizionario e converte le sentences negli interi corrispondenti del dizionario e pulisce le senteces dalle stop words
def createDictionary(sentences,path,max_words):
    tokenizer = Tokenizer(num_words=max_words)
	#assegna ad ogni token un indice
    tokenizer.fit_on_texts(sentences)
	#Restituiscele sentences divise in token e tradotte nei corrispondenti interi del dizionario
    sentences = tokenizer.texts_to_sequences(sentences)
    sentences = np.asarray(sentences)
    dictionary = tokenizer.word_index
    stop_words = set(stopwords.words('english'))
    my_file = Path(path)
    with open(path,'w') as dictionary_file:
        json.dump(dictionary,dictionary_file)
    filtered_sentence=list()
    for l in sentences:
        list_temp=list()
        for token in l:
            temp=token
            y=list(dictionary.keys())[list(dictionary.values()).index(token)]
            if not y in stop_words:
                list_temp.append(temp)
        filtered_sentence.append(list_temp)
    return dictionary,filtered_sentence

#converte le labels in interi e fa ritornare le label convertite, e le uniche in ordine come testo
def convert(labels):
    x = pd.Series(labels)
    s_enc_labels = pd.factorize(x)
    labels = s_enc_labels[0]
    labels = np.asarray(labels)
    #salva gli oggetti su un file
    return labels,s_enc_labels[1]

#ritorna la lunghezza della frase più lunga
def maxLenSentences(sentences):
    maxlen = 0
    for x in sentences:
	    if len(x)>maxlen:
		    maxlen=len(x)
    return maxlen

def load_word_embedding(embedding,embedding_path,dictionary,EMBED_SIZE):
    embedding_matrix = np.zeros((len(dictionary)+1, EMBED_SIZE))
    if embedding=='word2vec':
        import gensim
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=False)
        for word, index in dictionary.items():
            try:
                embedding_matrix[index, :] = word2vec[word]
            except KeyError:pass
    elif embedding=='glove':
        embeddings_index = {}
        f = open(embedding_path,encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        for word, i in dictionary.items():
	        embedding_vector = embeddings_index.get(word)
	        if embedding_vector is not None:
		        embedding_matrix[i] = embedding_vector
    return embedding_matrix,EMBED_SIZE

#prepara i dati per l'etichettatura
def prepareData(dictionary_word,data):
    words = kpt.text_to_word_sequence(data)
    wordIndices = []
    for word in words:
        if word in dictionary_word:
            wordIndices.append(dictionary_word[word])    
    return wordIndices


#scrive su un file le senteces, e le due emozioni etichettate
def writeFileTest(path,senteces,labels_1,labels_2):
    df = pd.DataFrame({'SIT':senteces,'Field1':labels_1,'Field2':labels_2})
    writer = pd.ExcelWriter(path)
    df.to_excel(writer,'Sheet1',index=False)
    writer.save()

#estrae le due emozioni dalla predizione	
def etichettatura(predizione,lista_sent,labels,path):
    etichette_1 = []
    etichette_2 = []
    for e in predizione:
        lista = list(e)
        max_1 = max(lista)
        etichette_1.append(labels[lista.index(max_1)])
        lista.remove(max_1)
        max_2 = max(lista)
        etichette_2.append(labels[lista.index(max_2)])
    writeFileTest(path,lista_sent,etichette_1,etichette_2)

#-----------------Metodi per il pos tagging--------------------#

#mi restituisce una lista in cui ogni elemento è a sua volta una lista formata dalla coppia token,pos
def IOB(sentences):
    info = list()
    for sent in sentences:
        token = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(token)
        info.append(tagged)
    return info	

#restituisce una lista di parole uniche e una lista di pos tag unici
def list_unique(doppia):
    pos = []
    words = []
    for sent in doppia:
        for idx, annotated_token in enumerate(sent):
            word, tag = annotated_token
            pos.append(tag)
            words.append(word)
    lista_unique_pos = list(set(pos))
    lista_unique_word = list(set(words))
    return lista_unique_pos,lista_unique_word

#crea il dizionario	delle parole,dei pos, dei ner
def createDictionaryTokenPos(path_word,path_pos,lista_word,lista_pos):
    pos_dictionary = {t: i+1 for i, t in enumerate(lista_pos)}
    word_dictionary = {t: i+1 for i,t in enumerate(lista_word)}
    with open(path_pos,'w') as dictionary_file_pos:
        json.dump(pos_dictionary,dictionary_file_pos)
    with open(path_word,'w') as dictionary_file_word:
        json.dump(word_dictionary,dictionary_file_word)
    return word_dictionary,pos_dictionary
	
#converte i dati nei rispettivi interi, le labels negli interi, e le labels uniche text
def convertDate(doppia,diz_word,diz_pos,labels):
    token_pos = [[[diz_word[w[0]],diz_pos[w[1]]] for w in s] for s in doppia]
    labels,labels_unique_text = convert(labels)
    return token_pos,labels,labels_unique_text

#Meotdo che mi restituisce le sentences tradotte nei corrispettivi interi (token,pos)
def prepareDatePos(dictionary_word,dictionary_pos,senteces):
    doppia = IOB(senteces)
    X = [[[dictionary_word[w[0]],dictionary_pos[w[1]]] for w in s if w[0] in dictionary_word and w[1] in dictionary_pos] for s in doppia]
    return X

#Metodo che mi restituisce le frasi migliori
def confronto(path_raw,path_pos):
    df_raw = pd.read_excel(path_raw)
    df_pos = pd.read_excel(path_pos)
    matrix_raw = df_raw.as_matrix()
    matrix_pos = df_pos.as_matrix()
    labels1_raw = [x[0] for x in matrix_raw]
    labels2_raw = [x[1] for x in matrix_raw]
    labels1_pos = [x[0] for x in matrix_pos]
    labels2_pos = [x[1] for x in matrix_pos]
    sentences = [x[2] for x in matrix_raw]
    sentences_final = []
    labels_final = []
    for i in range(len(labels1_raw)):
        if labels1_raw[i] == labels1_pos[i]:
            label = labels1_raw[i]
            sentences_final.append(sentences[i])
            labels_final.append(label)
        elif labels1_raw[i] == labels2_pos[i]:
            label = labels1_raw[i]
            sentences_final.append(sentences[i])
            labels_final.append(label)
        elif labels2_raw[i] == labels1_pos[i]:
            label = labels2_raw[i]
            sentences_final.append(sentences[i])
            labels_final.append(label)
        elif labels2_raw[i] == labels2_pos[i]:
            label = labels2_raw[i]
            sentences_final.append(sentences[i])
            labels_final.append(label)			
    return sentences_final,labels_final

#Vedo tra i vari clusters quale ha il minor numero di frasi migliori e mi faccio restituire tale valore	
def numMinimoFrasiCluster(clust,sentences):
    lista_min = list()
    for cluster in clust:
        tot = 0
        for sent in sentences:
            if sent in clust[cluster]:
                tot +=1
        if tot !=0:
            lista_min.append(tot)
    return min(lista_min)

#Elimino da ogni cluster x_min frasi migliori facendomele restituire in un array e in un altro array le corispettive emozioni	
def deleteSentCluster(cluster,lista_sent_migliori,lista_emotion_migliori,min):
    lun = len(lista_sent_migliori)
    lista = list()
    lista_emotion = list()
    for clust in cluster:
        i = 0
        j = 0
        while i<lun and j < min: 
            if lista_sent_migliori[i] in cluster[clust]:
                lista.append(lista_sent_migliori[i])
                lista_emotion.append(lista_emotion_migliori[i])
                cluster.setdefault(clust,cluster.get(clust).remove(lista_sent_migliori[i]))
                j +=1
            i += 1
    return lista,lista_emotion

#Elimino da l'array che contiene le restanti frasi quelle migliori estratte dai clusters, facendomi restituire le frasi totali aggiornate
def deleteSentTot(lista_totale,lista_sent):
    for sent in lista_sent:
	    lista_totale.remove(sent)
    return lista_totale