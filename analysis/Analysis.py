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
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from calculations.TfidfEmbeddingVectoriser import TfidfEmbeddingVectoriser


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

##Feature Extraction (Word2Vec)
nps_comment_feat = Word2Vec(nps_comment_filtered)
w2v_model = dict(zip(nps_comment_feat.wv.index2word, nps_comment_feat.wv.syn0))
etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectoriser(w2v_model)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectoriser(w2v_model)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])


#Split to test and train data
x_tr, x_te, y_tr, y_te = train_test_split(nps_comment_feat, nps_df.Survey_Responses_Survey_Rating, test_size=0.15, random_state=101)

#Naive bayes model
clf = MultinomialNB().fit(x_tr, y_tr)
predicted= clf.predict(x_te)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_te, predicted))











print (nps_df)