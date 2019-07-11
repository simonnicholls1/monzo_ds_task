from tabulate import tabulate
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from data_access.EscalationDAO import EscalationDAO
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

escalation_dao = EscalationDAO(credentials)


#Basic text filtering setup
stop_words=set(stopwords.words("english"))
ps = PorterStemmer()

####################################### escalation

#Get data
escalation_df = escalation_dao.get_escalation_data()
escalation_df = escalation_df.dropna()
#escalation_comment = escalation_df.Survey_Responses_Survey_Comments

##Pre Process

#Tokenise
escalation_comment_tokenised = escalation_df.apply(lambda row: word_tokenize(row.Cops_Escalations_Comments), axis=1)

#Remove stop words, punctuation and stem word
escalation_comment_filtered=[]
for s in escalation_comment_tokenised:
    comment = []
    for w in s:
        if w not in stop_words and w not in string.punctuation and len(w) > 3:
            comment.append(ps.stem(w.lower()))
    escalation_comment_filtered.append(comment)



fdist = FreqDist([item for sublist in escalation_comment_filtered for item in sublist])
fdist.plot(30,cumulative=False)
plt.show()


# Create Dictionary
id2word = corpora.Dictionary(escalation_comment_filtered)
# Create Corpus
texts = escalation_comment_filtered
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
coherence_model_lda = CoherenceModel(model=lda_model, texts=escalation_comment_filtered, dictionary=id2word, coherence='c_v')
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
