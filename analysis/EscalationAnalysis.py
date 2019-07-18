from tabulate import tabulate
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from data_access.EscalationDAO import EscalationDAO
from data_access.BigQueryConnection import BigQueryConnection
import matplotlib.pyplot as plt
from gensim.models import LdaModel
import string
from nltk.probability import FreqDist
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from nltk import ngrams
import re
import pandas as pd
from analysis.TextPreProcess import TextPreProcess

#Setup connection and data classes
big_query = BigQueryConnection()
credentials = big_query.get_credentials()

escalation_dao = EscalationDAO(credentials)


#Basic text filtering setup
stop_words=set(stopwords.words("english"))
ps = PorterStemmer()
text_filter = TextPreProcess()

####################################### Escalation

#Get data
escalation_df = escalation_dao.get_escalation_data()
escalation_df = escalation_df.dropna()
#escalation_df = pd.read_pickle('escalation.pickle')

#escalation_comment = escalation_df.Survey_Responses_Survey_Comments

################################Pre Process###################################################
#Tokenise
escalation_comment_tokenised = escalation_df.apply(lambda row: word_tokenize(row.Cops_Escalations_Comments), axis=1)
escalation_comment_sentence_tokenised = escalation_df.apply(lambda row: sent_tokenize(row.Cops_Escalations_Comments), axis=1)

#Remove stop words, punctuation and stem word
escalation_comment_filtered=[]
bi_grams = []
tri_grams = []
quad_grams = []
five_grams=[]
stop_words.add('customer')
stop_words.add('monzo')

[escalation_comment_filtered,bi_grams,tri_grams,quad_grams,five_grams, scores] = text_filter.filter_token_text(escalation_comment_tokenised, stop_words,None, False)

would_like_words = [key[2] for key in [item for sublist in tri_grams for item in sublist] if key[0] == 'would' and key[1]=='like']
would_like_abl_words = [key[3] for key in [item for sublist in quad_grams for item in sublist] if key[0] == 'would' and key[1]=='like' and key[2]=='abl']
would_like_abl_gram = [(key[3], key[4]) for key in [item for sublist in five_grams for item in sublist] if key[0] == 'would' and key[1]=='like' and key[2]=='abl']


################################Basic Distribution###################################################

#plot single word frequency
fdist = FreqDist([item for sublist in escalation_comment_filtered for item in sublist])
fdist.plot(30,cumulative=False)
plt.show()
plt.tight_layout()

#Plot bi grams
fdist_bi = FreqDist([item for sublist in bi_grams for item in sublist])
fdist_bi.plot(30,cumulative=False)
plt.show()
plt.tight_layout()

#Plot tri grams
fdist_tri = FreqDist([item for sublist in tri_grams for item in sublist])
fdist_tri.plot(30,cumulative=False)
plt.show()
plt.tight_layout()

#Plot would like
fdist_wl = FreqDist(would_like_words)
fdist_wl.plot(30,cumulative=False)
plt.show()
plt.tight_layout()

#Plot would like to be able to
fdist_wla = FreqDist(would_like_abl_words)
fdist_wla.plot(30,cumulative=False)
plt.show()
plt.tight_layout()

#Plot would like to be able to gram
fdist_wlag = FreqDist(would_like_abl_gram)
fdist_wlag.plot(30,cumulative=False)
plt.show()
plt.tight_layout()


#Remove anything with less than frequency of 5 or top 5 listed words also
escalation_comment_high_freq = []
high_freq_words = [key for key in fdist.keys() if fdist.get(key) > 4 and key not in [item[0] for item in list(fdist.most_common(5))]]
for s in escalation_comment_filtered:
    comment = []
    for w in s:
        if w in high_freq_words:
            comment.append(w)
        escalation_comment_high_freq.append(comment)
escalation_comment_filtered = escalation_comment_high_freq

###############################################Topic Modell1ing ##############################

# Create Dictionary
id2word = corpora.Dictionary(escalation_comment_filtered)
# Create Corpus
texts = escalation_comment_filtered
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
corpus_bigram = [id2word.doc2bow(text) for text in bi_grams]

#Create LDA model
lda_model = LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=4,
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
#coherence_model_lda = CoherenceModel(model=lda_model, texts=escalation_comment_filtered, dictionary=id2word, coherence='c_v')
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



###############################################Topic Modell1ing Bi Gram ##############################

# Create Dictionary
bi_gram_list = []
for s in bi_grams:
    comment = []
    for gram in s:
        text = gram[0] + '_' + gram[1]
        if text is not 'would_like':
            comment.append(gram[0] + '_'+ gram[1])
    bi_gram_list.append(comment)

id2word = corpora.Dictionary(bi_gram_list)
# Create Corpus
texts = escalation_comment_filtered
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in bi_gram_list]

#Create LDA model
lda_model = LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=4,
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
#coherence_model_lda = CoherenceModel(model=lda_model, texts=escalation_comment_filtered, dictionary=id2word, coherence='c_v')
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






