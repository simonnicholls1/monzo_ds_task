from tabulate import tabulate
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from data_access.NPSDAO import NPSDAO
from data_access.EscalationDAO import EscalationDAO
from data_access.CSATDAO import CSATDAO
from data_access.BigQueryConnection import BigQueryConnection
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, LdaModel
from calculations.MeanEmbeddingVectoriser import MeanEmbeddingVectoriser
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from calculations.TfidfEmbeddingVectoriser import TfidfEmbeddingVectoriser
from nltk.probability import FreqDist
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import pandas as pd
from collections import Counter


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

################################Pre Process###################################################

#Tokenise
nps_comment_tokenised = nps_df.apply(lambda row: word_tokenize(row.Survey_Responses_Survey_Comments), axis=1)

#Remove stop words, punctuation and stem word
nps_comment_filtered=[]
for s in nps_comment_tokenised:
    comment = []
    for w in s:
        if w not in stop_words and w not in string.punctuation and len(w) > 3:
            #comment.append(ps.stem(w.lower()))
            comment.append(w.lower())
    nps_comment_filtered.append(comment)


###########################Basic Distributions###############################################

fdist = FreqDist([item for sublist in nps_comment_filtered for item in sublist])
fdist.plot(30,cumulative=False)
plt.show()
'''

rating_count=nps_df.groupby('Survey_Responses_Survey_Rating').count()
plt.bar(rating_count.index.values, rating_count['Survey_Responses_Timestamp_Date'])
plt.xlabel('Ratings')
plt.ylabel('Number of Reviews')
plt.show()


nps_df['Date'] = nps_df.Survey_Responses_Timestamp_Date
avg_rating_per_day = nps_df.groupby('Date').mean()
plt.bar(avg_rating_per_day.index.values, avg_rating_per_day['Survey_Responses_Survey_Rating'])
plt.xlabel('Avg Rating')
plt.ylabel('Date')
plt.show()
'''
###############################################Topic Modelling ######################################################

# Create Dictionary
id2word = corpora.Dictionary(nps_comment_filtered)
# Create Corpus
texts = nps_comment_filtered
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=10,
                       random_state=100,
                       update_every=1,
                       chunksize=100,
                       passes=10,
                       alpha='auto',
                       per_word_topics=True)

# Print the Keyword in the 10 topics
print(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=lda_model, texts=nps_comment_filtered, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


data_flat =[w for w_list in nps_comment_filtered for w in w_list]
counter = Counter(data_flat)
out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 5, figsize=(20,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
plt.show()

###################################################Sentiment analysis#################################################

#Feature Extraction (Word2Vec trained on internal corpus)
model = Word2Vec(nps_comment_filtered, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

#Define all the different types of model some using mean some using tfid
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectoriser(w2v)), ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectoriser(w2v)), ("extra trees", ExtraTreesClassifier(n_estimators=200))])

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

#Run and display results of models
unsorted_scores = [(name, cross_val_score(model, nps_comment_filtered, nps_df.Survey_Responses_Survey_Rating, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])
print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
