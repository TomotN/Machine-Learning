from tokenizer import preprocess
import operator 
import json
from collections import Counter
from nltk.corpus import stopwords
import string
from collections import defaultdict
import math
from nltk import ngrams

fname = 'data_dir/stream_cats.json'
search_word = 'mouse'
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']
com = defaultdict(lambda : defaultdict(int))


positive_vocab = [
	'good', 'nice', 'great', 'awesome', 'fantastic', 'best', 'happy', 'soft' , 'play', 'better'
	]
negative_vocab = [
	'bad', 'terrible', 'crap', 'shit', 'blood', 'broken', 'ruin', 'hate'
	]


with open(fname, 'r') as f:
    count_stop = Counter()
    count_hash = Counter()
    count_search = Counter()
    count_ngrams = Counter()
    n_grams = 2
    for line in f:
        tweet = json.loads(line)
        # Create a list with all the terms
        terms_all = [term for term in preprocess(tweet['text'])]
        # Create a list with stop words removed
        terms_stop = [term for term in preprocess(tweet['text']) if term not in stop if len(term) > 2]
        # Update the counter
        terms_hash = [term for term in preprocess(tweet['text']) if term.startswith('#') if len(term) > 2]
        
        for i in range(len(terms_stop) -1):
        	for j in range(i+1, len(terms_stop)):
        		w1, w2 = sorted([terms_stop[i], terms_stop[j]])
        		if w1 != w2:
        			com[w1][w2] += 1
        if search_word in terms_stop:	# Note that this does not preclude the hashtags
        	count_search.update(terms_stop)
        
        grams = ngrams(tweet['text'].split(), n_grams)
        count_ngrams.update(grams)
        count_stop.update(terms_stop)
        count_hash.update(terms_hash)

    # Print the first 5 most frequent words
    print("\nCounters of tokenized words and hashtags:")
    print(count_stop.most_common(7))
    print(count_hash.most_common(7))
    print(count_ngrams.most_common(7))
    n_docs = len(terms_stop)
    com_max = []
    for t1 in com:
    	t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    	for t2, t2_count in t1_max_terms:
    		com_max.append(((t1, t2), t2_count))
    # Get the most frequent co-occurrences
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    print(terms_max[:7])

    print("\nCo-occurrence for %s:" % search_word)
    print(count_search.most_common(20))


    # Need to change the structure to numpy array?
    p_t = {}
    p_t_com = defaultdict(lambda : defaultdict(int))
    for term, n in count_stop.items():
    	p_t[term] = n / n_docs
    	for t2 in com[term]:
    		p_t_com[term][t2] = com[term][t2] / n_docs

    pmi = defaultdict(lambda : defaultdict(int))
    for t1 in p_t:
    	for t2 in com[t1]:
    		denom = p_t[t1] * p_t[t2]
    		pmi[t1][t2] = math.log(p_t_com[t1][t2] / denom)

    semantic_orientation = {}
    for term, n in p_t.items():
    	positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
    	negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
    	semantic_orientation[term] = positive_assoc - negative_assoc

    semantic_sorted = sorted(semantic_orientation.items(),key=operator.itemgetter(1), reverse=True)
    top_pos = semantic_sorted[:10]
    top_neg = semantic_sorted[-10:]

    print("\nPositve:", top_pos)
    print("\nNegative:", top_neg)
    # print('\nSemantic Sorted:', semantic_sorted)
    print("\nSemanic orientation of word:", semantic_orientation['fur'])
    print("\n")















