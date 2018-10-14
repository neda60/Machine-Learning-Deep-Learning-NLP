import numpy as np
import glob
import os
#import numpy.ma as ma
#from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from keras.layers import GRU
from keras.models import Sequential
from keras.layers import Dense, Embedding, Masking
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model
from keras import metrics
import keras
import pydot
from matplotlib import pyplot
import time



start_time = time.time()
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
nb_classes = 124 #182
nb_features= 7355 #   4019   #len(np.unique(train_data)) # extracting number of available features( functioons)
#nb_classes = 182
#nb_features= 4019   #len(np.unique(train_data)) # extracting number of available features( functioons)
savePath = r'I:\NEDA\My Journal\LSTM\Paper data'#r'C:\Users\umroot\Documents\Visual Studio 2015\Projects\deepLearning\deepLearning\binary_data'
dataSet = 'F' # 'G'  F for Firefox and G for Gnome
#name = 'F_11'   #model Name
name = dataSet + '_13'   #model Name


def load_data():
    max_len= -1
    train_data =[]
    train_labels =[]
    test_data =[]
    test_labels =[]
    seq = []
    path = r'I:\NEDA\FIREFOX DATA\HMM_J\HMMJ 124 stacks'#'I:\NEDA\GNOME\Converted to ID' #
    file = open(r'I:\NEDA\FIREFOX DATA\HMM_J\HMMJ 124 stacks\python_LSTM_labels.txt','w') # 'I:\NEDA\GNOME\HMMJ with unique traces\Gnome 182\python_LSTM_labels.txt'
    #path = r'I:\NEDA\GNOME\HMMJ with unique traces\Gnome 182'#'I:\NEDA\GNOME\Converted to ID' #
    #file = open(r'I:\NEDA\GNOME\HMMJ with unique traces\Gnome 182\python_LSTM_labels.txt','w') # 'I:\NEDA\GNOME\HMMJ with unique traces\Gnome 182\python_LSTM_labels.txt'
    lbl = -1 # to convert the labels to integers starting from 0

    #converte and save sequences in separate train and test files

    for filename in os.listdir(path + r'\Train'):
        with open(os.path.join(path + r'\Train', filename)) as f:
            lbl+=1
            file.write(filename + " " + str(lbl) + "\n")
            for line in f:
                seq = line.strip().split()
                train_data.append(seq)
                train_labels.append(lbl) #filename.split(".")[0]
                if max_len < len(line.split()):
                    max_len = len(line.split())

        with open(os.path.join(path+ r'\Validation', filename)) as f:
            for line in f:
                seq = line.strip().split()
                train_data.append(seq)
                train_labels.append(lbl) #filename.split(".")[0]
                if max_len < len(line.split()):
                    max_len = len(line.split())

        with open(os.path.join(path + r'\Test', filename)) as f:
            for line in f:
                seq = line.strip().split()
                test_data.append(seq)
                test_labels.append(lbl) #filename.split(".")[0]
                if max_len < len(line.split()):
                    max_len = len(line.split())
        
    nb_classes = lbl+1
    file.close()

    train_data = np.array(train_data)

    train_data = pad_sequences(train_data, maxlen = max_len, padding='post') 
    test_data = pad_sequences(test_data, maxlen= max_len, padding='post') 

    np.save(savePath + r'\\' + dataSet + 'train_data',train_data)
    np.save(savePath + r'\\' + dataSet + 'train_labels',train_labels)
    np.save(savePath + r'\\' + dataSet + 'test_data',test_data)
    np.save(savePath + r'\\' + dataSet + 'test_labels',test_labels)
    np.save(savePath + r'\\' + dataSet + 'max_len',max_len)


    return (np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)  , max_len, nb_classes) #, nb_features

if not os.path.exists(savePath + '\\' + name + '.h5'):

    if not os.path.exists(savePath + r'\\' + dataSet + 'train_data.npy'):
        train_data, train_labels, test_data, test_labels, max_len, nb_classes = load_data() #, nb_features
    else:
        train_data = np.load(savePath+ r'\\' + dataSet + 'train_data.npy')
        train_labels = np.load(savePath+ r'\\' + dataSet + 'train_labels.npy')
        test_data = np.load(savePath+ r'\\' + dataSet + 'test_data.npy')
        test_labels = np.load(savePath+ r'\\' + dataSet + 'test_labels.npy')
        max_len = np.load(savePath+ r'\\' + dataSet + 'max_len.npy')


    model = Sequential()
    model.add(Embedding(nb_features+1, 256, input_length = max_len, mask_zero=True))
    model.add(LSTM(256, return_sequences=True, dropout =0.5))# , recurrent_dropout =0.2)
    #model.add(Dropout(0.5))
    model.add(GRU(256, dropout =0.5))# , recurrent_dropout =0.2)
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu')) # could eliminate
    model.add(Dense(nb_classes, activation='softmax'))


    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy']) #added***
    # fit model


    history = model.fit(train_data, train_labels, epochs=30, batch_size=16, validation_data=(test_data, test_labels)) #validation_data=(x_test, y_test) # history = added***
    score = model.evaluate(test_data, test_labels,batch_size=8 ) # 
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


    model.save(savePath + '\\' + name+ ".h5")
    plot_model(model, to_file=name+'.png', show_shapes=True, show_layer_names=True)
    #pyplot.plot(history.history['accuracy']) 
    #pyplot.savefig(savePath + '\\' + name + '.png')#pyplot.show()
    result = model.predict(test_data)
    np.save(savePath+r'\\' + name + 'result',result)

else:
    model = keras.models.load_model(savePath + '\\' + name + '.h5')


    test_labels = np.load(savePath + r'\\' + dataSet + 'test_labels.npy')    
    test_data = np.load(savePath + r'\\' + dataSet + 'test_data.npy')
    #result = model.predict(test_data)
    result = np.load(savePath+r'\\' + name +  'result.npy')
    


def topKTotal(true_labeles,predicted,k=5):
    assert true_labeles.shape[0] == predicted.shape[0]
    count = 0
    for i in range(true_labeles.shape[0]):
        args = np.argsort(predicted[i])
        if true_labeles[i] in args[-k:]:
            count += 1
    return count/true_labeles.shape[0]

def topKClasses(true_labeles,predicted,nb_classes,k=5):
    assert true_labeles.shape[0] == predicted.shape[0]
    countTrue = np.zeros(nb_classes)
    countAll = np.zeros(nb_classes)
    for i in range(true_labeles.shape[0]):
        args = np.argsort(predicted[i])
        countAll[true_labeles[i]] +=1
        if true_labeles[i] in args[-k:]:
            countTrue[true_labeles[i]] +=1
    return countTrue/countAll

txt = open(savePath + '\\' + name + 'topK_ALL.txt','w')
for i in range(1,21):
    res1 = topKTotal(test_labels, result, i)
    res2 = topKClasses(test_labels, result, nb_classes, i)
    np.savetxt(savePath + '\\' + name + 'top' + np.str(i) + 'K_Classes', np.array(res2))
    txt.write(np.str(i) + '\t'+ np.str(res1) + '\n')

txt.close()

print(time.time()-start_time)# calculating the execution time
