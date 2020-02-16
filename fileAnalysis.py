
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
    
    def __init__(self,max_clusters, train_corpus, isKBuilt = False):


        
        self.training_corpus = train_corpus
        self.tfidfVec()
        self.mnb = self.MultinomialNB()
        self.kmeans_model(max_clusters,isKBuilt)
        self.true_clusters = self.kmeans.n_clusters
        self.cv = self.CountVectorizer()
        self.bayes_model()


    """
    CRITICAL FOR LATER
    THE KMEANS WILL ONLY WORK IF APPLIED TO DIFFERENT DATA
    RANDOMIZE ARTICLE SELECTION IN ACTUAL RUNTHROUGH
    """
    
    def tfidfVec(self):
        self.tfidf_vectorizer = self.TfidfVectorizer().fit(self.training_corpus)
        self.tfidf = self.tfidf_vectorizer.transform(self.training_corpus)
        size = len(self.tfidf.toarray()//2)
        self.score_sample = self.tfidf_vectorizer.fit_transform(self.random.sample(self.training_corpus, size))
        self.train_sample = self.tfidf_vectorizer.fit_transform(self.random.sample(self.training_corpus, size))

        self.tf_array = self.tfidf.toarray()

        

   
    def kmeans_model(self, clusters_range, built = False):
        
        
        kmeans_file = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers\kmeans_model.pickle"
        
        if(built == False):
        
            kmeans = [self.KMeans(n_clusters = i, algorithm = "full").fit(self.train_sample) for i in range(1,clusters_range)]
            score = [kmeans[i].fit(self.train_sample).score(self.score_sample) for i in range(len(kmeans))]
        
            self.plt.plot(range(1,clusters_range), score)
            self.plt.xlabel('Number of Clusters')
            self.plt.ylabel('Score')
            self.plt.title('Elbow Method')
            self.plt.show()
            self.true_clusters = int(input("elbow point"))
            self.kmeans = self.KMeans(n_clusters = self.true_clusters, algorithm = "full",random_state = 0, init = 'random').fit(self.tfidf)
            self.getDescriptors()
            self.getPCA()
            
            with open(kmeans_file,'wb') as file:
                print("---- BUILDING KMEANS MODEL ----")
                self.pickle.dump(self.kmeans,file)
                file.close()
        else:
            with open(kmeans_file, "rb") as file:
                print("----LOADING KMEANS MODEL---")
                self.kmeans = self.pickle.load(file)
                file.close()
                

    def getDescriptors(self):
        print("Top terms per cluster:")
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = self.tfidf_vectorizer.get_feature_names()
        for i in range(self.true_clusters):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :20]:
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
    NEED TO HAVE A RANDOM SAMPLE OF DOCUMENTS
    """
    def prep_for_bayes(self):
        training_dict = {}
        for document_pair in zip(self.training_corpus,self.kmeans.labels_):
            cluster = document_pair[1]
            training_dict[document_pair[0]] = (cluster)
        class_associations = self.pd.DataFrame.from_dict(training_dict, orient = "index", columns = ['labels'])
        class_associations = class_associations.rename_axis("articles").reset_index()
        return class_associations
        
  
    def bayes_model(self):
        
        split_set = self.prep_for_bayes()
        self.cv = self.cv.fit(split_set["articles"])
        random_x = split_set.sample(frac = .5)
        random_y = split_set.sample(frac = .5)
        half = int(len(random_x)/2)
        
        train_x = random_x["articles"][:half] 
        test_x = random_x["articles"][half:]
        
        train_y = random_y["labels"][:half] 
        test_y =  random_y["labels"][half:] 
        
        train_x_counts = self.cv.transform(train_x)
        self.mnb = self.mnb.fit(train_x_counts,train_y)
        
        self.test_x_counts = self.cv.transform(test_x)
        predicted_x = self.mnb.predict(self.test_x_counts)
        score = self.metrics.accuracy_score(test_y,predicted_x)
        print(score)

    def bayes_classify(self, sentence_token):
        predict_array = self.cv.transform(sentence_token)
        return self.mnb.predict(predict_array)




from fileProcessing import fileProcess

file = "D:\SeniorProject\FakeCorGazReorganized\FakeCorGaz18990901.xml"
root_FC = r"D:\SeniorProject\FakeCorGaz"
target_FC = r"D:\SeniorProject\FakeCorGazReorganized"
fp_FC = fileProcess(root_FC,target_FC, sample_size=10, news_paper= "FakeCorGaz",isCorpusBuilt = True)
fa_FC = fileAnalysis(max_clusters=10,train_corpus=fp_FC.corpus,isKBuilt=True)

df = fa_FC.generateDF(fp_FC.df)





