#import and prepare data
ocr_text="output extracted text daal do"

#import the library and initialize teh spell checker
from spellchecker import SpellChecker
spell= SpellChecker() #default to english
#tokenize the text into words
words= ocr_text.split()

#remove(strip) Punctuation
import string
clean_words = [w.strip(string.punctuation) for w in words]

#Run the correction
corrected_words=[spell.correction(word) for word in clean_words]

#reconstruct the sentence
corrected_sentence=" ".join(corrected_words)
print(corrected_sentence)
