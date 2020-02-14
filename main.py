from fileProcessing import fileProcess
from fileAnalysis import fileAnalysis


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
Corpus Objects

"""

fp_Cor = fileProcess(root_Cor,target_Cor, "CorGaz")
fp_RD = fileProcess(root_RD,target_RD, "RightsDem")

"""
RUN ONCE (RAN 2/13/20)

"""
fp_Cor.walkAndProcess()
fp_RD.walkAndProcess()


"""
FOR SAMPLING PURPOSES:
COR GAZ: 730 issues
RIGHTS DEM: 
10% of each for corpus sample
Random sample of both for training?
"""
#CODE FOR CORPUS


#fa = fileAnalysis(fp,clusters = 10,sample_size = 20,isCorpusBuilt = True, isKBuilt = True)

#CODE FOR MODEL BUILDING
#build model on random sample of CorGaz, or make it work for sample of both


