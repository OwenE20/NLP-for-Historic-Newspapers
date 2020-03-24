

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
Corpus Objects

"""
from fileProcessing import fileProcess

train_path = r"D:\SeniorProject\TrainingFiles"
root = r"D:\SeniorProject\RawData\TrainingSNs"




#sample size can be huge (any portion of total)
fp_Training = fileProcess(root,train_path,"TrainingFiles",sample_size= 4400, isFileSetup = True, isCorpusBuilt=True)
    
#sample size can be 700
fp_Cor = fileProcess(root_Cor,target_Cor, "CorGaz",sample_size=700, isFileSetup = True, isCorpusBuilt=True)

#sample size can be 1200
fp_RD = fileProcess(root_RD,target_RD, "RightsDem",sample_size=1200,isFileSetup = True,isCorpusBuilt=True)


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
model = fileAnalysis(max_clusters=10,train_corpus=Train_Corpus, stop_words = fp_Training.stopset, isKBuilt = False,isBayesBuilt = False)

"""
Once model is built, get DF from both Corpora
"""


#Cor_df = model.generateDF(Cor_Corpus)
#RD_df = model.generateDF(RD_Corpus)

"""
Once dfs are built, collect statistics
ALSO, GET A VIGNETTE

"""

#%%
#import pandas as pd

#Cor_df = model.generateDF(Cor_Corpus)
#RD_df = model.generateDF(RD_Corpus)

"""
Cor_proportions = Cor_df
RD_proportions = RD_df

print("n of RD: " + str(RD_df["num_articles"].sum()))
print("n of Cor: " + str(Cor_df["num_articles"].sum()))
for column in Cor_proportions.columns[2:]:
    prop = Cor_df[column].sum() / Cor_df["num_articles"].sum()
    print(str(column) + " proportion of Cor: + " + str(prop))
    Cor_proportions[column] = Cor_df[column] / Cor_df["num_articles"]
    print("stats of Cor column: " + str(column)  + str(Cor_proportions[column].describe()))


for column in (RD_proportions.columns[2:]):
    prop = RD_df[column].sum() / RD_df["num_articles"].sum()
    print(str(column) + " proportion of RD: + " + str(prop))
    RD_proportions[column] = RD_df[column] / RD_df["num_articles"]
    print("stats of RD column: " + str(column)  + str(RD_proportions[column].describe()))


RD_y = RD_proportions.columns[2:].astype("str")

Cor_y = Cor_proportions.columns[2:].astype("str")

"""


#Cor_proportions = Cor_proportions.sort_index()
#RD_proportions = RD_proportions.sort_index()
#Cor_proportions.to_csv(r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers\CorGaz.csv")
#RD_proportions.to_csv(r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers\RD.csv")





def random_training_set():
    import os
    import random
    import shutil

    #FIVE PERCENT OF EACH WILL SERVE FOR TRAINING

    five_percent_CD = .1 * len(os.listdir(target_Cor))
    five_percent_RD = .1 * len(os.listdir(target_RD))

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


#random_training_set()

