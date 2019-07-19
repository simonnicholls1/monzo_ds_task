from tabulate import tabulate
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
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
from analysis.TextPreProcess import TextPreProcess
from collections import Counter
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Setup connection and data classes
big_query = BigQueryConnection()
credentials = big_query.get_credentials()

csat_dao = CSATDAO(credentials)

#Basic text filtering setup
stop_words=set(stopwords.words("english"))
stop_words.add('help')
stop_words.add('helpful')
stop_words.add('monzo')

####################################### CSAT



#Get data
csat_df = csat_dao.get_csat_data()
csat_df = csat_df.dropna()

###########################Basic Sentiment###################################################
#Do some basic sentiment analysis here using Vader, will train own model later

nltk_sentiment = SentimentIntensityAnalyzer()
csat_df['vader_sentiment'] = csat_df.Conversations_Remark.apply(nltk_sentiment.polarity_scores)

#csat_comment = csat_df.Conversations_Remark

################################Pre Process###################################################

csat_df['Number_Of_Words'] = csat_df.Conversations_Remark.str.split().apply(len)
csat_df.reset_index(inplace=True)

#Tokenise
csat_comment_tokenised = csat_df.apply(lambda row: word_tokenize(row.Conversations_Remark), axis=1)
#Get data per rating also
csat_comment_tokenised_1 = csat_comment_tokenised.loc[csat_df.Conversations_Rating == 1]
csat_comment_tokenised_2 = csat_comment_tokenised.loc[csat_df.Conversations_Rating == 2]
csat_comment_tokenised_3 = csat_comment_tokenised.loc[csat_df.Conversations_Rating == 3]
csat_comment_tokenised_4 = csat_comment_tokenised.loc[csat_df.Conversations_Rating == 4]
csat_comment_tokenised_5 = csat_comment_tokenised.loc[csat_df.Conversations_Rating == 5]



#Filter
text_filter = TextPreProcess()
[csat_comment_filtered, bi_grams, tri_grams, four_grams, five_grams,score_return] = text_filter.filter_token_text(csat_comment_tokenised, stop_words,csat_df.Conversations_Rating, False)

[csat_comment_filtered_1, bi_grams_1, tri_grams_1, four_grams_1, five_grams_1,score_return_1] = text_filter.filter_token_text(csat_comment_tokenised_1, stop_words)
[csat_comment_filtered_2, bi_grams_2, tri_grams_2, four_grams_2, five_grams_2,score_return_2] = text_filter.filter_token_text(csat_comment_tokenised_2, stop_words)
[csat_comment_filtered_3, bi_grams_3, tri_grams_3, four_grams_3, five_grams_3,score_return_3] = text_filter.filter_token_text(csat_comment_tokenised_3, stop_words)
[csat_comment_filtered_4, bi_grams_4, tri_grams_4, four_grams_4, five_grams_4,score_return_4] = text_filter.filter_token_text(csat_comment_tokenised_4, stop_words)
[csat_comment_filtered_5, bi_grams_5, tri_grams_5, four_grams_5, five_grams_5,score_return_5] = text_filter.filter_token_text(csat_comment_tokenised_5, stop_words)


###########################Basic Distributions###############################################

csat_df.groupby('Conversations_Rating').count().Conversations_Remark.plot.bar(x = 'Rating', y='Count')
#Distribution of scores
for i in [csat_comment_filtered, csat_comment_filtered_1, csat_comment_filtered_2, csat_comment_filtered_3, csat_comment_filtered_4, csat_comment_filtered_5]:
    fdist = FreqDist([item for sublist in i for item in sublist])
    fdist.plot(30,cumulative=False)
    plt.tight_layout()
    plt.show()
'''

##Create word cloud for each rating
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=40,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

rating_list = [csat_comment_filtered, csat_comment_filtered_1, csat_comment_filtered_2, csat_comment_filtered_3, csat_comment_filtered_4, csat_comment_filtered_5]
fig, axes = plt.subplots(3, 2, figsize=(10,10), sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    words = dict(FreqDist([item for sublist in rating_list[i] for item in sublist]))
    cloud.generate_from_frequencies(words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Rating ' + str(i+1), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

#No of Reviews per rating
rating_count=csat_df.groupby('Conversations_Rating').count()
plt.bar(rating_count.index.values, rating_count['Conversations_Conversation_Start_Date'])
plt.xlabel('Ratings')
plt.ylabel('Distribution of Rating')
plt.show()



#Rating per date
csat_df['Date'] = csat_df.Conversations_Conversation_Start_Date
avg_rating_per_day = csat_df.groupby('Date').mean()
plt.bar(avg_rating_per_day.index.values, avg_rating_per_day['Conversations_Rating'])
plt.xlabel('Avg Rating')
plt.ylabel('Date')
plt.title('CSAT - Average Rating Per Date')
plt.show()

csat_df['Date'] = csat_df.Conversations_Conversation_Start_Date
count_rating_per_day = csat_df.groupby('Date').count()
plt.bar(avg_rating_per_day.index.values, count_rating_per_day['Conversations_Remark'])
plt.xlabel('Count')
plt.ylabel('Date')
plt.title('CSAT - Count Per Date')
plt.show()

#Avg words per rating
avg_no_words_per_rating = csat_df.groupby('Conversations_Rating').mean()
plt.bar(avg_no_words_per_rating.index.values, avg_no_words_per_rating['Number_Of_Words'])
plt.xlabel('Avg words')
plt.ylabel('Rating')
plt.show()

###############################################Topic Modell1ing ######################################################

# Create Dictionary
id2word = corpora.Dictionary(csat_comment_filtered)
# Create Corpus
texts = csat_comment_filtered
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

#Create LDA model
lda_model = LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=12,
                       random_state=100,
                       update_every=1,
                       chunksize=100,
                       passes=10,
                       alpha='auto',
                       per_word_topics=True)

# Print the Keyword in the 10 topics
print(lda_model.print_topics(num_topics=12))
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=csat_comment_filtered, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

##Create word cloud for each topic
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=30,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)
topics = lda_model.show_topics(formatted=False, num_topics=12)

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

data_flat =[w for w_list in csat_comment_filtered for w in w_list]
counter = Counter(data_flat)
out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(10,10), sharey=True, dpi=160)
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
'''
###################################################Sentiment analysis#################################################

#Feature Extraction (Word2Vec trained on internal corpus)
model = Word2Vec(csat_comment_filtered, size=100, window=5, min_count=5, workers=2)
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
unsorted_scores = [(name, cross_val_score(model, csat_comment_filtered, score_return, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])
print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
