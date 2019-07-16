import string
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker

class TextPreProcess:

    def __init__(self):
        self.ps = PorterStemmer()
        self.spell = SpellChecker()
    def filter_token_text(self, tokenised_text, stop_words, spell_check = False):
        # Remove stop words, punctuation and stem word
        csat_comment_filtered = []
        for s in tokenised_text:
            comment = []
            for w in s:
                if w not in stop_words and w not in string.punctuation and len(w) > 3:
                    text = self.ps.stem(w.lower())
                    if(spell_check):
                        text = self.spell.correction(text)
                    comment.append(text)
            if len(comment) > 0:
                csat_comment_filtered.append(comment)

        return csat_comment_filtered