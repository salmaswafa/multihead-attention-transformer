# from nltk.tokenize import word_tokenize
# import nltk

# # Ensure punkt is downloaded
# nltk.download('punkt')

# # Test tokenization
# text = "Hello, world! This is a test."
# tokens = word_tokenize(text)
# print(tokens)

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

text = "Good muffins cost $3.88 in New York. Please buy me two of them. Thanks."
tokens = word_tokenize(text)
print(tokens)