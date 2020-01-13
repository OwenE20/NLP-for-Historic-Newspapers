
class fileAnalysis:
    from os import listdir
    from os.path import isfile,join
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    
    import os
    
    def __init__(self, fileprocess):
        self.fp = fileprocess
        self.corpus = self.buildCorpus()
        self.tfidfVec()
       
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
     
    """
    CRITICAL FOR LATER
    THE KMEANS WILL ONLY WORK IF APPLIED TO DIFFERENT DATA
    RANDOMIZE ARTICLE SELECTION IN ACTUAL RUNTHROUGH
    """
    
    def tfidfVec(self):
        self.tfidf_vectorizer = self.TfidfVectorizer()
        self.tfidf = self.tfidf_vectorizer.fit_transform(self.corpus[:100])

   
    def kmeans_model(self, clusters_range):
        kmeans = [self.KMeans(n_clusters = i, algorithm = "full").fit(self.tfidf) for i in range(1,clusters_range)]
        score = [kmeans[i].fit(self.tfidf).score(self.tfidf) for i in range(len(kmeans))]
        
        self.plt.plot(range(1,clusters_range), score)
        self.plt.xlabel('Number of Clusters')
        self.plt.ylabel('Score')
        self.plt.title('Elbow Method')
        self.plt.show()
        
        self.true_clusters = int(input("elbow point"))
        k_means = self.KMeans(n_clusters = self.true_clusters, algorithm = "full").fit(self.tfidf)
        return k_means
    
    def getDescriptors(self,kmeans):
        print("Top terms per cluster:")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = self.tfidf_vectorizer.get_feature_names()
        for i in range(self.true_clusters):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
                
    def getPCA(self,kmeans):
        svd = self.TruncatedSVD()
        svd_points = svd.fit_transform(self.tfidf.toarray())
        kmeans = self.KMeans(n_clusters= self.true_clusters, max_iter=600, algorithm = 'full')
        fitted = kmeans.fit(svd_points)
        prediction = kmeans.predict(svd_points)
        
        self.plt.scatter(svd_points[:, 0], svd_points[:, 1], c=prediction, s=50, cmap='viridis')
        centers = fitted.cluster_centers_
        self.plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
        self.plt.show()
        
        
    def generateDF(self):
        self.df = df.rename(columns = {0:"articles"})
        self.df["num_articles"] =self.pd.Series([len(self.df["articles"][i]) for i in range(0,len(self.df["articles"]))], index = self.df.index)
        for clusters in true_clusters:
            frequencies = []
            for index,row in self.df.iterrows():
                frequency = 0
                for article in self.df.at[index,"articles"]:
                    if(kmeans.predict(article))
                
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
kmeans = fa.kmeans_model(20)
fa.getDescriptors(kmeans)
fa.getPCA(kmeans)

