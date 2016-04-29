from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import pandas as pd

# Create a list of each user's job title (i.e. a "corpus" of documents)
corpus = [
    "underwriter manager",
    "mortgage data analyst",
    "junior underwriter",
    
    "sales manager",
    "junior medical sales associate",
    "senior sales manager",
    "sales manager",
    
    "senior data analyst",
    "data analyst",
    "data manager",
    
    "data analyst manager",
    "junior data analyst"
]

categories = [
    "finance",
    "finance",
    "finance",
    
    "sales",
    "sales",
    "sales",
    "sales",
    
    "technology",
    "technology",
    "technology",
    
    "",
    ""
]

# Convert the categories Technology, Sales, and Finance to numbers 0, 1, and 2
y = list(map(lambda x: {'finance':0, 'sales':1, 'technology':2}[x], categories[:10]))

# Buid an MXN matrix where M is the number of samples, N is the number of unique words in the corpus (subject to parameters) and element [i,j] is the
# whether or not sample i contains word j
count_vectorizer = CountVectorizer()
count_vectorizer.fit(corpus)
X = count_vectorizer.transform(corpus[:10])

# Dump results into a pandas DataFrame since this is a small example for illustrative purposes
df = pd.DataFrame(X.toarray(), columns=count_vectorizer.get_feature_names())
df

# Now consider a new title
X_new = count_vectorizer.transform(corpus[10:12])
df_new = pd.DataFrame(X_new.toarray(), columns=count_vectorizer.get_feature_names())
df_new

# Check our results with scikit-learn's Bernoulli Naive Bayes classifier
naive_bayes = BernoulliNB(alpha=0.000000001)  # make alpha virtually 0
naive_bayes.fit(X=df, y=y)
naive_bayes.predict_proba(X=df_new)

# Re-run with alpha = 1
naive_bayes = BernoulliNB(alpha=1)  # make alpha 1
naive_bayes.fit(X=df, y=y)
naive_bayes.predict_proba(X=df_new)
