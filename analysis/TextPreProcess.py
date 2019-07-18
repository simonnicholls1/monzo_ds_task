import string
from nltk.stem import PorterStemmer
from autocorrect import spell
from nltk import ngrams
import re

class TextPreProcess:

    def __init__(self):
        self.ps = PorterStemmer()

    def filter_token_text(self, tokenised_text, stop_words, scores = None, spell_check = False):
        # Remove stop words, punctuation and stem word
        csat_comment_filtered = []
        bi_grams = []
        tri_grams = []
        quad_grams = []
        five_grams = []
        score_return = []
        for i, s in enumerate(tokenised_text):
            comment = []
            for w in s:
                text = w.lower()
                if text not in stop_words and text not in string.punctuation and len(text) > 3:
                    text = re.sub('[^0-9a-zA-Z]+', '', w)
                    if (spell_check):
                        text = spell(text.lower())
                    text = self.ps.stem(text)
                    comment.append(text)
            if len(comment) > 0:
                bi_grams.append(list(ngrams(comment, 2)))
                tri_grams.append(list(ngrams(comment, 3)))
                quad_grams.append(list(ngrams(comment, 4)))
                five_grams.append(list(ngrams(comment, 5)))
                csat_comment_filtered.append(comment)
                if scores is not None:
                    score_return.append(scores[i])

        return csat_comment_filtered,bi_grams,tri_grams,quad_grams,five_grams,score_return