
class fileAnalysis:
    from os import listdir
    from os.path import isfile,join
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    
    def __init__(self , fileprocess):
        self.fp = fileprocess
        self.corpus_document = []
    
    def buildCorpus(self):
        fileList = []
        for file in listdir(fp.target):
            fileList.append(join(fp.target,file))
        self.df = fp.move_to_df(fileList)
        corpus = []
        for index, data in self.df.iterrows():
            corpus += data[0]
        return corpus
    
    def kmeans_model(self,corpus):
        tfidf_vectorizer = self.TfidfVectorizer()
        tfidf = tfidf_vectorizer(corpus)
        
        
        
    
"""
weighted = TfidfVectorizer().fit_transform(df.iat[0,0])
k = KMeans(n_clusters = 4).fit(weighted)

clusters = pd.DataFrame(k.cluster_centers_)

terms = weighted.
"""

from fileProcessing import fileProcess

root_dir = r"D:\SeniorProject\testDir"
target_dir = r"D:\SeniorProject\CorGazReorganized"
news_name = "CorGaz" 

files = ["D:\SeniorProject\CorGazReorganized/CorGaz18991027.xml","D:\SeniorProject\CorGazReorganized\CorGaz18990922.xml"]



fp = fileProcess(root_dir,target_dir, "CorGaz")
fa = fileAnalysis(fp)

from os import listdir
from os.path import isfile,join
cor = fa.buildCorpus()









