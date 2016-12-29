import nltk
import numpy as np
import json
from tokenizer import preprocess
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords

########################################################################
#   Config Information
########################################################################
fname = 'data_dir/stream_cats.json'
np.set_printoptions(threshold=np.inf)
wordnet_lemmatizer = WordNetLemmatizer()

punctuation = list(string.punctuation)
stopwords = stopwords.words('english') + punctuation + ['rt', 'via', '@']
########################################################################
#   Functions
########################################################################

# Input Matricies for ANN
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label

    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label
    return x

# Creates the polarised lists containing pre determined information
def get_polarised_lists():
    with open(fname, 'r') as f:

        positive = []
        negative = []
        word_index_map = {}
        current_index = 0


        for line in f:
            tweet = json.loads(line)

            token_list = []
            token_list = [term for term in preprocess(tweet['text']) if term not in stopwords if len(term) > 2]
            # print(token_list)
            if 'love' in token_list:
                # count_pos.update(tweet['text'])
                data = ' '.join(token_list)+'\n'
                positive.append(token_list)
                for token in token_list:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                # with open(pos_outfile, 'a') as f:
                #     f.write(data)
            elif 'hate' in token_list:
                data = ' '.join(token_list)+'\n'
                negative.append(token_list)
                for token in token_list:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                # with open(neg_outfile, 'a') as f:
                #     f.write(data)
        return positive, negative, word_index_map


########################################################################
#   Main Body
########################################################################
# so let's take a random sample so we have balanced classes

positive, negative, word_index_map = get_polarised_lists()
np.random.shuffle(positive)
positive = positive[:len(negative)]
print ('Number in positive list:',len(positive))
print(positive[0:3])    # Lists of tokenized tweets. 

N = len(positive) + len(negative)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1


np.random.shuffle(data)
# print data[:,1]


X = data[:, :-1]
Y = data[:, -1]

Xtrain = X[:-50,]
Ytrain = Y[:-50,]
Xtest  = X[-50:,]
Ytest  = Y[-50:,]

# print ("Xtest", wXtest[0:5])
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print ("Classification rate:", model.score(Xtest, Ytest))


# let's look at the weights for each word
threshold = 0.5
for word, index in word_index_map.iteritems():
    weight = model.coef_[0][index]
    if abs(weight) > threshold:
        print word, weight
    














