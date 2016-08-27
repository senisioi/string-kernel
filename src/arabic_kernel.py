import os
import sys
import codecs
import scipy
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.svm import SVC

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)


KERN_SIZE = 5
NFOLD = 5

'''
    The number of common character n-grams
'''
def normalized_kern_sum(str1, str2, n=3):
    s1 = set([str1[i:i + n] for i in range(len(str1) - n + 1)])
    s2 = set([str2[i:i + n] for i in range(len(str2) - n + 1)])
    return len(s1 & s2) * 1/np.sqrt(len(s1) * len(s2)) 

'''
    The number of common character n-grams with n = [q,p]
'''
def normalized_kern_spectrum(str1, str2, p=5):
    s1 = set()
    START = 2
    if (p <= START):
        logging.error("The spectrum size has to be larger than 2. Aborting...")
        sys.exit(1)
    for q in range(START, p):
        s1 = s1.union( set([str1[i:i + q] for i in range(len(str1) - q + 1)]) )
    s2 = set()
    for q in range(START, p):
        s2 = s2.union( set([str2[i:i + q] for i in range(len(str2) - q + 1)]) )

    if len(s1) == 0 or len(s2) == 0:
        return 0
    
    return len(s1 & s2) * 1/np.sqrt(len(s1) * len(s2)) 

def pairwise_kernel_train(list_of_text, kernel, subseq_length, unique_id=''):
    kernel_dump = './kernel_train_'+ kernel.__name__+ '_'+ str(subseq_length)+'_'+ os.path.basename(unique_id)+'_.pkl'
    ################################################################################
    cntr = 0
    total = len(list_of_text) * (len(list_of_text) - 1) / 2
    retval = np.ones((len(list_of_text), len(list_of_text)))
    for i in range(0, len(list_of_text)-1):
        for j in range(i+1, len(list_of_text)):
            cntr += 1
            if (total - cntr) % 100000 == 0:
                logging.info('Remaining '+ str(total - cntr)) 
            retval[i, j] = kernel(list_of_text[i], list_of_text[j], subseq_length) 
            retval[j, i] = retval[i, j]
    retval[np.isnan(retval)] = 0
    ################################################################################    
    with open(kernel_dump, 'wb') as fout:
        pickle.dump(retval, fout)

    return retval

def pairwise_kernel_test(test_data, train_data, kernel, subseq_length, unique_id=''):
    kernel_dump = './kernel_test_'+ kernel.__name__+ '_'+ str(subseq_length)+'_'+ os.path.basename(unique_id)+'.pkl'
    ################################################################################
    cntr = 0
    total = len(test_data) * len(train_data) 
    retval = np.ones((len(test_data), len(train_data)))
    for i in range(0, len(test_data)):
        for j in range(0, len(train_data)):
            cntr += 1
            if (total - cntr) % 100000 == 0:
                logging.info('Remaining '+ str(total - cntr)) 
            retval[i, j] = kernel(test_data[i], train_data[j], subseq_length) 
    retval[np.isnan(retval)] = 0
    ################################################################################
    with open(kernel_dump, 'wb') as fout:
        pickle.dump(retval, fout)

    return retval

def pairwise_kernel(test_data, train_data, subseq_length, unique_id):
    logging.info('Computing kernel...')
    kernel = normalized_kern_spectrum
    if np.array_equal(test_data, train_data):
        return pairwise_kernel_train(train_data, kernel, subseq_length, unique_id)
    else:
        return pairwise_kernel_test(test_data, train_data, kernel, subseq_length, unique_id)

def train_SVM_kernel(train_kernel, y, test_kernel):
    svm_model = SVC(kernel='precomputed')
    svm_model.fit(train_kernel, y)
    logging.info('Predicting...')
    return svm_model.predict(test_kernel)
    logging.info('Done!')


def main():
    data = pd.read_csv("../data/arabic.csv", '\t')

    lab2num = dict()
    num2lab = dict()
    for i,l in enumerate(set(data["labels"])):
        lab2num[l] = i
        num2lab[i] = l

    skf = StratifiedKFold(NFOLD, shuffle=True)
    cv_splits = skf.split(data["sentence"], data["labels"])

    accuracy_scores = []

    counter = 0
    for train,test in cv_splits:
        logging.info('Computing Train Kern...')
        tr_kern = data['sentence'][train].values
        y = data['labels'][train].replace(lab2num).values
        unique_id = str(counter)+"_train_"
        train_kernel = pairwise_kernel(tr_kern, tr_kern, KERN_SIZE, unique_id)
        with open("Y_"+ str(KERN_SIZE) + "_" + str(counter) + "_train.pkl", 'wb') as fout:
            pickle.dump(y, fout)
        
        logging.info('Computing Test Kern...')
        ts_kern = data['sentence'][test].values
        y_test = data['labels'][test].replace(lab2num).values
        unique_id = str(counter)+"_test_"
        test_kernel = pairwise_kernel(ts_kern, tr_kern, KERN_SIZE, unique_id)
        with open("Y_"+ str(KERN_SIZE) + "_" + str(counter) + "_test.pkl", 'wb') as fout:
            pickle.dump(data['labels'][test].replace(lab2num).values, fout)
        
        counter += 1

        svmker = train_SVM_kernel(train_kernel, y, test_kernel)
        score = metrics.accuracy_score(y_test, svmker) 
        #appending all the scores
        with open("scores", "a") as score_out:
            score_out.write(str(score) + "\n")
            
        accuracy_scores.append(score)
        print score

    print str(accuracy_scores)
    print np.mean(accuracy_scores)

if __name__ == '__main__':
    main()
