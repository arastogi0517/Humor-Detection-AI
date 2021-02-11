import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Read in pickle files
humor = pd.read_pickle('datasets/humorous_oneliners.pickle')
proverb = pd.read_pickle('datasets/proverbs.pickle')
wiki = pd.read_pickle('datasets/wiki_sentences.pickle')
reuters = pd.read_pickle('datasets/reuters_headlines.pickle')

# Classify as funny or not funny
humor_record = [(sentence, 1) for sentence in humor]
proverb_record = [(sentence, 0) for sentence in proverb]
wiki_record = [(sentence, 0) for sentence in wiki]
reuter_record = [(sentence, 0) for sentence in reuters]

# Put funny sentences and not funny sentences together
pos_record = humor_record
neg_record = proverb_record + wiki_record + reuter_record

# Since there are fewer funny sentences, randomly choose len(funny) sentences
# from the non-funny sentences to train the model
random.shuffle(neg_record)
neg_record = neg_record[:len(pos_record)]

# Construct and shuffle dataframe
columns = ['sentence', 'classifications']
df_record = pos_record + neg_record
df = pd.DataFrame(df_record, columns=columns)
df = df.sample(frac=1).reset_index(drop=True)

# Use Word2Vec to map words to vectors
text=[]
for pair in df_record:
    sentence = pair[0]
    sent_word_list = [word for word in sentence.lower().split()]
    text.append(sent_word_list)

w2v = Word2Vec(text, min_count=1)

# Construct a list of the sentence vectors
vect_record=[]
for i in range(len(df['sentence'])):
    sent = df['sentence'][i]
    if len(sent)!=0:
        sent_vect = sum([w2v.wv[w] for w in sent.lower().split()])/(len(sent.split())+0.001)
    else:
        sent_vect = np.zeros((100,))
    vect_record.append(sent_vect)

# Construct a dataframe to store the vectors
X = pd.DataFrame(vect_record, columns=range(100))
y = df['classifications']

# Performaing train test split (80% training data, 20% testing data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Use Support Vector Machine Classifier and compute accuracy
svm = SVC(kernel='linear', random_state=1, C=1.0, probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('Test Accuracy Score: ', accuracy_score(y_pred, y_test))

'''
# Cross-validation
scores = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))
'''

'''
# Use Logistic Regression Classifier and compute accuracy
lr = LogisticRegression(C=100.0, random_state=1, max_iter = 1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('Test Accuracy score: ', accuracy_score(y_pred, y_test))

# Cross-validation
scores = cross_val_score(estimator=lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))
'''

# Receive input and assign funniness score
joke = input('Enter a joke! ')
while (joke != 'q'):
    joke_word_list = [word for word in joke.lower().split()]
    text.append(joke_word_list)
    w2v = Word2Vec(text, min_count=1)
    joke_vect = sum([w2v.wv[w] for w in joke.lower().split()])/(len(joke.split())+0.001)
    joke_np = pd.DataFrame(joke_vect).T

    funny_score = svm.predict_proba(joke_np)[0][1]

    if funny_score >= .8:
        print('Nice joke! Very funny! Your funny score was {:.2f}'.format(funny_score * 100))
    else:
        print('Sorry, that was not very funny. Try again! Your funny score was {:.2f}'.format(funny_score * 100))

    joke = input('Enter a joke! ')