import string
from nltk.stem import PorterStemmer

class TextPreProcess:

    def __init__(self):
        self.ps = PorterStemmer()

    def filter_token_text(self, tokenised_text, stop_words):
        # Remove stop words, punctuation and stem word
        csat_comment_filtered = []
        for s in tokenised_text:
            comment = []
            for w in s:
                if w not in stop_words and w not in string.punctuation and len(w) > 3:
                    comment.append(self.ps.stem(w.lower()))
            if len(comment) > 0:
                csat_comment_filtered.append(comment)

        return csat_comment_filtered