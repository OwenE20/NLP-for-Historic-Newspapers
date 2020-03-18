from fileProcessing import fileProcess
import sklearn.pipeline as p
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer,silhouette_score

train_path = r"D:\SeniorProject\TrainingFiles"
root = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers"

fp_Training = fileProcess(root,train_path,"TrainingFiles",sample_size= 200,isCorpusBuilt=False)

pipeline = p.Pipeline([
    ('tfidf', TfidfVectorizer())
])

parameters = {
    'tfidf_max_df': (.35, .45,.5,.75,1.0),
    'tfidf_min_df': (0,.02,.04,.06,.08),
    'tfidf_ngram_range': [(1,3)],\
    "tfidf_smooth_idf": (True,False),
    "tfidf_sublinear_tf": (True,False),
    "norm": ('l1','l2',None),
    "max_features" : (1000,5000,10000,15000,20000),
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
svd = TruncatedSVD()
for paramset in random.sample(experiments,k = 100):
    print(paramset)
    plt.clf()
    vec = TfidfVectorizer(
                            max_df= paramset["tfidf_max_df"],
                            min_df= paramset["tfidf_min_df"],
                            ngram_range= paramset["tfidf_ngram_range"],
                            use_idf= True,
                            smooth_idf = paramset["tfidf_smooth_idf"],
                            sublinear_tf= paramset["tfidf_sublinear_tf"],
                            norm = paramset["norm"],
                            max_features = paramset["max_features"],
                            analyzer="word",
                            binary= paramset["binary"]

    )
    
    tfidf = vec.fit_transform(fp_Training.corpus)
    print("num of dimensions in raw: %d" % len(tfidf.toarray()[0]))
    svd_points = svd.fit_transform(tfidf.toarray())
    plt.title(str(paramset) + " " + str(len(tfidf.toarray()[0])))
    plt.scatter(svd_points[:, 0], svd_points[:, 1], s=50, cmap='viridis')
    plt.show()



