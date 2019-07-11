from tabulate import tabulate
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from data_access.CSATDAO import CSATDAO
from data_access.BigQueryConnection import BigQueryConnection
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, LdaModel
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
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from wordcloud import WordCloud
import matplotlib.colors as mcolors




#Setup connection and data classes
big_query = BigQueryConnection()
credentials = big_query.get_credentials()

csat_dao = CSATDAO(credentials)

#Basic text filtering setup
stop_words=set(stopwords.words("english"))
ps = PorterStemmer()

####################################### csat

#Get data
csat_df = csat_dao.get_csat_data()
csat_df = csat_df.dropna()
#csat_comment = csat_df.Conversations_Remark

##Pre Process

csat_df['Number_Of_Words'] = csat_df.Conversations_Remark.str.split().apply(len)

#Tokenise
csat_comment_tokenised = csat_df.apply(lambda row: word_tokenize(row.Conversations_Remark), axis=1)

#Remove stop words, punctuation and stem word
csat_comment_filtered=[]
for s in csat_comment_tokenised:
    comment = []
    for w in s:
        if w not in stop_words and w not in string.punctuation and len(w) > 3:
            comment.append(ps.stem(w.lower()))
    csat_comment_filtered.append(comment)



fdist = FreqDist([item for sublist in csat_comment_filtered for item in sublist])
fdist.plot(30,cumulative=False)
plt.show()


rating_count=csat_df.groupby('Conversations_Rating').count()
plt.bar(rating_count.index.values, rating_count['Conversations_Conversation_Start_Date'])
plt.xlabel('Ratings')
plt.ylabel('Number of Reviews')
plt.show()


csat_df['Date'] = csat_df.Conversations_Conversation_Start_Date
avg_rating_per_day = csat_df.groupby('Date').mean()
plt.bar(avg_rating_per_day.index.values, avg_rating_per_day['Conversations_Rating'])
plt.xlabel('Avg Rating')
plt.ylabel('Date')
plt.show()

avg_no_words_per_rating = csat_df.groupby('Conversations_Rating').mean()
plt.bar(avg_no_words_per_rating.index.values, avg_no_words_per_rating['Number_Of_Words'])
plt.xlabel('Avg words')
plt.ylabel('Rating')
plt.show()


# Create Dictionary
id2word = corpora.Dictionary(csat_comment_filtered)
# Create Corpus
texts = csat_comment_filtered
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=20,
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
coherence_model_lda = CoherenceModel(model=lda_model, texts=csat_comment_filtered, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

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

##Feature Extraction (Word2Vec)

mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

model = Word2Vec(csat_comment_filtered, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}


# Extra Trees classifier is almost universally great, let's stack it with our embeddings
etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectoriser(w2v)),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectoriser(w2v)),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])



#Split to test and train data
vector_data = MeanEmbeddingVectoriser(w2v).transform(csat_comment_filtered)
x_tr, x_te, y_tr, y_te = train_test_split(vector_data, csat_df.Conversations_Rating, test_size=0.15, random_state=101)


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


unsorted_scores = [(name, cross_val_score(model, csat_comment_filtered, csat_df.Conversations_Rating, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])
print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
