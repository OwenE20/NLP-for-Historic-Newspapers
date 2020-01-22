
class fileAnalysis:
    from os import listdir
    from os.path import isfile,join
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    import os
    from sklearn import metrics
    import pickle
    import random
    
    """
    AFTER EVERYTHING IS TESTED AND WORKING, EVERY PARAMATER SHOULD BE AN ATTRIBUTE
    ALSO AFTER EVERYTHING IS WORKING, PICKLE PARAMETERS FOR MODELS, CORPUS, etc.
    """
    
    def __init__(self, fileprocess,clusters,sample_size = 2, isCorpusBuilt = False, isKBuilt = False):
        
        corpus_filename = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers\corpus.pickle"
        
        
        if(isCorpusBuilt == False):
            with open(corpus_filename,'wb') as file:
                print("---- BUILDING CORPUS ----")
                self.corpus = self.buildCorpus(sample_size)
                self.pickle.dump(self.corpus,file)
                file.close()
        else:
            with open(corpus_filename, "rb") as file:
                print("----LOADING CORPUS---")
                self.corpus = self.pickle.load(file)
                file.close()
                
                
        
        self.fp = fileprocess
        self.tfidfVec()
        self.mnb = self.MultinomialNB()
        self.kmeans_model(clusters,isKBuilt)
        self.cv = self.CountVectorizer()
       
    #sample_size is how many files
    def buildCorpus(self,sample_size):
        fileList = []
        for file in self.listdir(fp.target):
            fileList.append(self.os.path.join(fp.target,file))
        self.df = fp.move_to_df(fileList,sample_size)
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
        self.tfidf = self.tfidf_vectorizer.fit_transform(self.corpus)
        self.tf_array = self.tfidf.toarray()
        

   
    def kmeans_model(self, clusters_range, built = False):
        
        
        kmeans_file = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers\kmeans_model.pickle"
        
        if(built == False):
            sample_size =  len(self.tf_array//2)
            train_sample = self.random.sample(self.tf_array,sample_size)
            score_sample = self.random.sample(self.tf_array, sample_size)
            
        
            kmeans = [self.KMeans(n_clusters = i, algorithm = "full").fit(train_sample) for i in range(1,clusters_range)]
            score = [kmeans[i].fit(train_sample).score(score_sample) for i in range(len(kmeans))]
        
            self.plt.plot(range(1,clusters_range), score)
            self.plt.xlabel('Number of Clusters')
            self.plt.ylabel('Score')
            self.plt.title('Elbow Method')
            self.plt.show()
            self.true_clusters = int(input("elbow point"))
            self.kmeans = self.KMeans(n_clusters = self.true_clusters, algorithm = "full").fit(self.tfidf)
            
            with open(kmeans_file,'wb') as file:
                print("---- BUILDING KMEANS MODEL ----")
                self.pickle.dump(self.kmeans,file)
                file.close()
        else:
            with open(kmeans_file, "rb") as file:
                print("----LOADING KMEANS MODEL---")
                self.kmeans = self.pickle.load(file)
                file.close()
                
        self.getDescriptors()
        self.getPCA()
        
        
       
    def getDescriptors(self):
        print("Top terms per cluster:")
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = self.tfidf_vectorizer.get_feature_names()
        for i in range(self.true_clusters):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
                
    def getPCA(self):
        svd = self.TruncatedSVD()
        svd_points = svd.fit_transform(self.tfidf.toarray())
        fitted = self.kmeans.fit(svd_points)
        prediction = self.kmeans.predict(svd_points)
        
        self.plt.scatter(svd_points[:, 0], svd_points[:, 1], c=prediction, s=50, cmap='viridis')
        centers = fitted.cluster_centers_
        self.plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
        self.plt.show()
        
      
    """
    THIS WILL BE USEFUL TO GENERATE DATA AFTER BAYES MODEL IS TRAINED
     
     
    def generateDF(self,kmeans):
        self.df = self.df.rename(columns = {0:"articles"})
        self.df["num_articles"] =self.pd.Series([len(self.df["articles"][i]) for i in range(0,len(self.df["articles"]))], index = self.df.index)
        for clusters in range(self.true_clusters):
            frequencies = []
            for index,row in self.df.iterrows():
                frequency = 0
                for article in self.df.at[index,"articles"]:
                    if(kmeans.predict(self.tfidf_vectorizer.transform([article])[0]) == clusters):
                        frequency += 1
                frequencies.append(frequency)
            self.df["cluster " + str(clusters)] = self.pd.Series(frequency, index = self.df.index)
    """
           
    """
    NEED TO HAVE A RANDOM SAMPLE OF DOCUMENTS
    """
    def prep_for_bayes(self):
        training_dict = {}
        for document in self.corpus:
            cluster = self.kmeans.predict(self.tfidf_vectorizer.transform([document])[0])
            training_dict[document] = (cluster)
        class_associations = self.pd.DataFrame.from_dict(training_dict, orient = "index", columns = ['labels'])
        class_associations = class_associations.rename_axis("articles").reset_index()
        return class_associations
        
  
    def bayes_model(self):
        
        split_set = self.prep_for_bayes()
        random_x = split_set.sample(frac = .5)
        random_y = split_set.sample(frac = .5)
        half = int(len(random_x)/2)
        
        train_x = random_x["articles"][:half] 
        test_x = random_x["articles"][half:]
        
        train_y = random_y["labels"][:half] 
        test_y =  random_y["labels"][half:] 
        
        train_x_counts = self.cv.fit_transform(train_x)
        self.mnb = self.mnb.fit(train_x_counts,train_y)
        
        self.test_x_counts = self.cv.fit_transform(test_x)
        print(test_x_counts)
        predicted_x = self.mnb.predict(test_x_counts)
        print(predicted_x)
        score = self.metrics.accuracy_score(test_y,predicted_y)
        print(score)

from fileProcessing import fileProcess
root_dir = r"D:\SeniorProject\testDir"
target_dir = r"D:\SeniorProject\CorGazReorganized"
news_name = "CorGaz" 

files = ["D:\SeniorProject\CorGazReorganized/CorGaz18991027.xml","D:\SeniorProject\CorGazReorganized\CorGaz18990922.xml"]


"""

text,date = fp.parse_xml(files[1])
list2 = fp.cleanList(text)

"""
fp = fileProcess(root_dir,target_dir, "CorGaz")
fa = fileAnalysis(fp,clusters = 10,sample_size = 10,isCorpusBuilt = True, isKBuilt = False)

