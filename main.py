

root_Cor = r"D:\SeniorProject\CorGaz"
target_Cor = r"D:\SeniorProject\CorGazReorganized"

root_RD = r"D:\SeniorProject\RightsDem"
target_RD = r"D:\SeniorProject\RightsDemReorganized"


"""
RUN ONCE (RAN 2/13/20)
fp_Cor.walkAndProcess()
fp_RD.walkAndProcess()

"""

"""
For Training: took 5% from each dataset
Run once for training sample
Ran (2/15/2020)




def random_training_set():
    import os
    import random
    import shutil

    #FIVE PERCENT OF EACH WILL SERVE FOR TRAINING

    five_percent_CD = .05 * len(os.listdir(target_Cor))
    five_percent_RD = .05 * len(os.listdir(target_RD))

    train_num = 0
    while(train_num < five_percent_CD):
        file = os.path.join(target_Cor,random.choice(os.listdir(target_Cor)))
        shutil.move(file,train_path)
        train_num += 1

    train_num = 0
    while (train_num < five_percent_RD):
        file = os.path.join(target_RD, random.choice(os.listdir(target_RD)))
        shutil.move(file, train_path)
        train_num += 1
"""


"""
FOR SAMPLING PURPOSES:
COR GAZ: 689 files 
RIGHTS DEM: 1,264 files
Training
10% of each for corpus sample
Random sample of both for training?
"""


"""
Corpus Objects

"""
from fileProcessing import fileProcess

train_path = r"D:\SeniorProject\TrainingFiles"
root = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers"

#true sample = 104
fp_Training = fileProcess(root,train_path,"TrainingFiles",sample_size= 104,isCorpusBuilt=True)
fp_Cor = fileProcess(root_Cor,target_Cor, "CorGaz",sample_size=69, isCorpusBuilt=True)
fp_RD = fileProcess(root_RD,target_RD, "RightsDem",sample_size=126,isCorpusBuilt=True)


Cor_Corpus = fp_Cor.df
RD_Corpus = fp_RD.df
Train_Corpus = fp_Training.corpus


"""
Run once to reduce common words to increase difference between categories
(Ran 2/20/20)
"""

from collections import Counter
def generateStopSet(corpus):
    big_list = []
    for string in corpus:
        for word in list(string.split()):
            big_list.append(word)
    cs = Counter(big_list)
    hundred_most_common = dict(cs.most_common(200))
    return set(hundred_most_common.keys())


#stop_set = generateStopSet(Train_Corpus)
#stop_file = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers\stop.pickle"
#import pickle

#with open(stop_file, 'wb') as file:
 #   pickle.dump(stop_set, file)
  #  file.close()

from fileAnalysis import fileAnalysis
model = fileAnalysis(max_clusters=20,train_corpus=Train_Corpus,isKBuilt = False,isBayesBuilt = False)

"""
Once model is built, get DF from both Corpora

"""
#Cor_df = model.generateDF(Cor_Corpus)
#RD_df = model.generateDF(RD_Corpus)

"""
Once dfs are built, collect statistics

"""



