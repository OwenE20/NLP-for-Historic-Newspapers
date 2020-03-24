from fileProcessing import fileProcess
import sklearn.pipeline as p
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer,silhouette_score
from sklearn.utils.extmath import randomized_svd

train_path = r"D:\SeniorProject\TrainingFiles"
root = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers"

fp_Training = fileProcess(root,train_path,"TrainingFiles",sample_size= 4400, isFileSetup = True, isCorpusBuilt=True)

pipeline = p.Pipeline([
    ('tfidf', TfidfVectorizer())
])

parameters = {
    'tfidf_max_df': (.75,1.0),
    'tfidf_min_df': (0,.005,.01,.02,.04),
    'tfidf_ngram_range': [(1,3)],
    "norm": ('l1','l2'),
    "max_features" : (15000,20000,25000),
    "binary": (True,False)
}
"""
vectorizer = TfidfVectorizer()
from sklearn.decomposition import LatentDirichletAllocation
components = 10
dsa = LatentDirichletAllocation(n_components= components, n_jobs=1).fit(vectorizer.fit_transform(fp_Training.corpus))
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words -1:-1]]))
print_topics(dsa,vectorizer,20)


"""

import itertools
keys, values = zip(*parameters.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
import random
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
for paramset in random.sample(experiments,k = 50):
    print(paramset)
    plt.clf()
    vec = TfidfVectorizer(
                            max_df= paramset["tfidf_max_df"],
                            min_df= paramset["tfidf_min_df"],
                            ngram_range= paramset["tfidf_ngram_range"],
                            use_idf= True,
                            smooth_idf =True,
                            sublinear_tf= True,
                            norm = paramset["norm"],
                            max_features = paramset["max_features"],
                            analyzer="word",
                            binary= paramset["binary"]

    )
    
    tfidf = vec.fit_transform(fp_Training.corpus)
    
   
    print("num of dimensions in raw: %d" % len(tfidf.toarray()[0]))
    kmeans = [KMeans(n_clusters = i, algorithm = "full",).fit(tfidf) for i in range(2,5)]

    silhouette_scores = [silhouette_score(tfidf,kmeans[i].fit(tfidf).labels_) for i in range(0,len(kmeans))]
    plt.title(str(paramset) + " ")
    plt.plot(range(2,5), silhouette_scores)
    plt.show()



