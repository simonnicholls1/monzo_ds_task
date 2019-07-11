from tabulate import tabulate
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from data_access.NPSDAO import NPSDAO
from data_access.EscalationDAO import EscalationDAO
from data_access.CSATDAO import CSATDAO
from data_access.BigQueryConnection import BigQueryConnection
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from calculations.MeanEmbeddingVectoriser import MeanEmbeddingVectoriser
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from calculations.TfidfEmbeddingVectoriser import TfidfEmbeddingVectoriser
from nltk.probability import FreqDist


#Setup connection and data classes
big_query = BigQueryConnection()
credentials = big_query.get_credentials()

csat_dao = CSATDAO(credentials)
escalation_dao = EscalationDAO(credentials)
nps_dao = NPSDAO(credentials)

#Basic text filtering setup
stop_words=set(stopwords.words("english"))
ps = PorterStemmer()

####################################### NPS

#Get data
nps_df = nps_dao.get_nps_data()
nps_df = nps_df.dropna()
#nps_comment = nps_df.Survey_Responses_Survey_Comments

##Pre Process

#Tokenise
nps_comment_tokenised = nps_df.apply(lambda row: word_tokenize(row.Survey_Responses_Survey_Comments), axis=1)

#Remove stop words, punctuation and stem word
nps_comment_filtered=[]
for s in nps_comment_tokenised:
    comment = []
    for w in s:
        if w not in stop_words and w not in string.punctuation and len(w) > 3:
            comment.append(ps.stem(w.lower()))
    nps_comment_filtered.append(comment)



fdist = FreqDist([item for sublist in nps_comment_filtered for item in sublist])
fdist.plot(30,cumulative=False)
plt.show()


rating_count=nps_df.groupby('Survey_Responses_Survey_Rating').count()
plt.bar(rating_count.index.values, rating_count['Survey_Responses_Timestamp_Date'])
plt.xlabel('Ratings').resample('W-Mon', on='Date').resample('W-Mon', on='Date').resample('W-Mon', on='Date')
plt.ylabel('Number of Reviews')
plt.show()


nps_df['Date'] = nps_df.Survey_Responses_Timestamp_Date
avg_rating_per_day = nps_df.groupby('Date').mean()
plt.bar(avg_rating_per_day.index.values, avg_rating_per_day['Survey_Responses_Survey_Rating'])
plt.xlabel('Avg Rating')
plt.ylabel('Date')
plt.show()

##Feature Extraction (Word2Vec)


mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

model = Word2Vec(nps_comment_filtered, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}


# Extra Trees classifier is almost universally great, let's stack it with our embeddings
etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectoriser(w2v)),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectoriser(w2v)),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])



#Split to test and train data
vector_data = MeanEmbeddingVectoriser(w2v).transform(nps_comment_filtered)
x_tr, x_te, y_tr, y_te = train_test_split(vector_data, nps_df.Survey_Responses_Survey_Rating, test_size=0.15, random_state=101)


all_models = [
    ("mult_nb", mult_nb),
    ("mult_nb_tfidf", mult_nb_tfidf),
    ("bern_nb", bern_nb),
    ("bern_nb_tfidf", bern_nb_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf)

]


unsorted_scores = [(name, cross_val_score(model, nps_comment_filtered, nps_df.Survey_Responses_Survey_Rating, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])
print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
