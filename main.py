#from fileProcessing import fileProcess
#from fileAnalysis import fileAnalysis





"""
root_FC = r"D:\SeniorProject\FakeCorGaz"
target_FC = r"D:\SeniorProject\FakeCorGazReorganized"
fp_FC = fileProcess(root_FC,target_FC, "FakeCorGaz")
fp_FC.walkAndProcess();


"""

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

"""

def random_training_set():
    import os
    import random
    import shutil

    #FIVE PERCENT OF EACH WILL SERVE FOR TRAINING

    train_path = r"D:\SeniorProject\TrainingFiles"
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
FOR SAMPLING PURPOSES:
COR GAZ: 689 files 
RIGHTS DEM: 1,264 files
10% of each for corpus sample
Random sample of both for training?
"""


"""
Corpus Objects

"""

#fp_Cor = fileProcess(root_Cor,target_Cor, "CorGaz")
#fp_RD = fileProcess(root_RD,target_RD, "RightsDem")




#CODE FOR CORPUS


#fa = fileAnalysis(fp,clusters = 10,sample_size = 20,isCorpusBuilt = True, isKBuilt = True)

#CODE FOR MODEL BUILDING
#build model on random sample of CorGaz, or make it work for sample of both


