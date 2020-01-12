
class fileAnalysis:
    from os import listdir
    from os.path import isfile,join
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    import os
    
    def __init__(self , fileprocess):
        self.fp = fileprocess
        self.corpus_document = []

        
    
    def buildCorpus(self):
        fileList = []
        for file in self.listdir(fp.target):
            fileList.append(self.os.path.join(fp.target,file))
        self.df = fp.move_to_df(fileList)
        corpus = []
        for index, data in self.df.iterrows():
            for index, element in enumerate(data[0]):
                corpus.append(element)
        return corpus
     
        
   
    def kmeans_model(self,corpus,clusters_range):
        self.tfidf_vectorizer = self.TfidfVectorizer()
        self.tfidf = self.tfidf_vectorizer.fit_transform(corpus[:100])

        kmeans = [self.KMeans(n_clusters = i, algorithm = "full").fit(self.tfidf) for i in range(1,clusters_range)]
        score = [kmeans[i].fit(self.tfidf).score(self.tfidf) for i in range(len(kmeans))]
        
        self.plt.plot(range(1,clusters_range), score)
        self.plt.xlabel('Number of Clusters')
        self.plt.ylabel('Score')
        self.plt.title('Elbow Method')
        self.plt.show()
        
        true_clusters = input("elbow point")
        print(true_clusters)
        k_means = self.KMeans(n_clusters = int(true_clusters), algorithm = "full").fit(self.tfidf)
        return k_means
    
    def getDescriptors(self,kmeans):
        print("Top terms per cluster:")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = self.tfidf_vectorizer.get_feature_names()
        for i in range(2):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
                
    """
    def graphPCA(self,kmeans)
    """
           
        
        
        
    
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
cor = fa.buildCorpus()
kmeans = fa.kmeans_model(cor,30)
fa.getDescriptors(kmeans)





