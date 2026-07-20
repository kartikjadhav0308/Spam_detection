import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
# cv = CountVectorizer()
tfid = TfidfVectorizer(max_features=3000)
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# Download resources once
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
#read data
df = pd.read_csv(r"DataSet\spam.csv",encoding = "latin1")

#drop the last column which having the more null values
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

#Rename columns
df.rename(columns={'v1':"target",'v2':'text'},inplace=True)

#encoding the spam and not spam column

encoder = LabelEncoder()
df['target']=encoder.fit_transform(df['target'])

#DROP DUPLICATED VALUES
# print(df.duplicated().sum())
df = df.drop_duplicates(keep='first')
# print(df.duplicated().sum())
# print(df.shape)

#EDA

# print(df['target'].value_counts())
df['num_characters']=df['text'].apply(len)

df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

#Data preprocessing  1.Lower case 2.tokenization 3.Removing special characters 4.Removing stop words and punctuation 5.Stemming

def transform_text(text):
    text = text.lower()
    text= nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
df['transformed_text']=df['text'].apply(transform_text)


spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

#make the dictionary
Counter(spam_corpus).most_common(30)

ham_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

#Model Building
X = tfid.fit_transform(df['transformed_text']).toarray()
y = df['target'].values


X_train,X_test,y_train,y_text = train_test_split(X,y,test_size=0.2,random_state=2)

# print("\nGaussianNB:\n")
# gnb.fit(X_train,y_train)
# y_pred1 = gnb.predict(X_test)
# print(accuracy_score(y_text,y_pred1))
# print(confusion_matrix(y_text,y_pred1))
# print(precision_score(y_text,y_pred1))

# print("\nBernoulliNB:\n")
# bnb.fit(X_train,y_train)
# y_pred2 = bnb.predict(X_test)
# print(accuracy_score(y_text,y_pred2))
# print(confusion_matrix(y_text,y_pred2))
# print(precision_score(y_text,y_pred2))

# print("\nMultinomialNB:\n")
mnb.fit(X_train,y_train)
# y_pred3 = mnb.predict(X_test)
# print(accuracy_score(y_text,y_pred3))
# print(confusion_matrix(y_text,y_pred3))
# print(precision_score(y_text,y_pred3))

#we are use the multinomialNB which has more pricision and accuracy



# Paste your email/message here
email_text = input("Paste Your Email Below:\n")

# Preprocess the email
transformed_email = transform_text(email_text)

# Convert text into TF-IDF vector
email_vector = tfid.transform([transformed_email]).toarray()

# Predict
prediction = mnb.predict(email_vector)

# Display result
if prediction[0] == 1:
    print("Spam Email 🚨")
else:
    print("Not Spam Email ✅")



#URGENT! You have won $1000. Send your bank details to receive your reward. ->spam
"""
Hi Kartik,
Can we schedule a meeting tomorrow at 10 AM?
Thanks.  ->not spam
"""