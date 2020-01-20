
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
    #from sklearn.feature_extraction import CountVectorizer
    from sklearn.model_selection import train_test_split
    import os
    import pickle
    
    """
    AFTER EVERYTHING IS TESTED AND WORKING, EVERY PARAMATER SHOULD BE AN ATTRIBUTE
    ALSO AFTER EVERYTHING IS WORKING, PICKLE PARAMETERS FOR MODELS, CORPUS, etc.
    """
    
    def __init__(self, fileprocess, isCorpusBuilt = False):
        
        corpus_filename = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers\corpus.pickle"
        corpus_outfile = open(corpus_filename,'wb')
        if(isCorpusBuilt == False):
            print("---- BUILDING CORPUS ----")
            self.corpus = self.buildCorpus()
            print(len(self.corpus))
            self.pickle.dump(self.corpus,corpus_outfile)
            corpus_outfile.close()
        else:
            corpus_infile = open(corpus_filename,"rb")
            print("----LOADING CORPUS---")
            self.corpus = self.pickle.load(corpus_infile)
            corpus_infile.close()
            print(self.corpus[0])
            
        self.fp = fileprocess
        self.tfidfVec()
        self.mnb = self.MultinomialNB()
        #self.cv = self.CountVectorizer()
       
    #Make corpus a sample
    def buildCorpus(self):
        fileList = []
        for file in self.listdir(fp.target)[:1]:
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
        self.tfidf = self.tfidf_vectorizer.fit_transform(self.corpus[:5])

   
    def kmeans_model(self, clusters_range):
        """
        kmeans = [self.KMeans(n_clusters = i, algorithm = "full").fit(self.tfidf) for i in range(1,clusters_range)]
        score = [kmeans[i].fit(self.tfidf).score(self.tfidf) for i in range(len(kmeans))]
        
        self.plt.plot(range(1,clusters_range), score)
        self.plt.xlabel('Number of Clusters')
        self.plt.ylabel('Score')
        self.plt.title('Elbow Method')
        self.plt.show()
        """
        #self.true_clusters = int(input("elbow point"))
        self.true_clusters = clusters_range
        k_means = self.KMeans(n_clusters = 2, algorithm = "full").fit(self.tfidf)
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
    def prep_for_bayes(self, kmeans):
        training_dict = {}
        for document in self.corpus[:6]:
            cluster = kmeans.predict(self.tfidf_vectorizer.transform([document])[0])
            training_dict[document] = (cluster)
        class_associations = self.pd.DataFrame.from_dict(training_dict, orient = "index", columns = ['articles'])
        print("new")
        return class_associations
        
  
    def bayes_model(self,kmeans):
        #train_x = self.randomize_df(self.prep_for_bayes(kmeans))
        #text_x = self.randomize_df(self.prep_for_bayes(kmeans))
        #train_y = self.randomize_df(self.prep_for_bayes(kmeans))
        #text_y = self.randomize_df(self.prep_for_bayes(kmeans))
        train_x, test_x, train_y, test_y = self.train_test_split(self.prep_for_bayes(kmeans).index,self.prep_for_bayes()["0"])
        print(train_x,train_y,test_y,test_y)
        
    """  
     def randomize_df(self,series,sample_size):
        return series.sample(n = sample_size, random_state = 1)
    """
                


from fileProcessing import fileProcess


root_dir = r"D:\SeniorProject\testDir"
target_dir = r"D:\SeniorProject\CorGazReorganized"
news_name = "CorGaz" 

files = ["D:\SeniorProject\CorGazReorganized/CorGaz18991027.xml","D:\SeniorProject\CorGazReorganized\CorGaz18990922.xml"]

fp = fileProcess(root_dir,target_dir, "CorGaz")
fa = fileAnalysis(fp,isCorpusBuilt = False)
"""

text,date = fp.parse_xml(files[1])
list2 = fp.cleanList(text)

"""

kmeans = fa.kmeans_model(10)
pd = fa.prep_for_bayes(kmeans)


#%%




#fa.generateDF(kmeans)
#df = fa.df

#fa.getDescriptors(kmeans)
#fa.getPCA(kmeans)

