from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Step 1
emails = fetch_20newsgroups()
print(emails.target_names)

# Step 2
categories = ['rec.sport.baseball', 'rec.sport.hockey']
emails = fetch_20newsgroups(categories=categories)

# Step 3
print(emails.data[5])

# Step 4 (Prints the label: 0 or 1)
print(emails.target[5])
print('\nThe email is about:', emails.target_names[emails.target[5]], "\n")

# Step 5
train_emails = fetch_20newsgroups(categories=categories, subset='train', shuffle=True, random_state=108)

# Step 6
test_emails = fetch_20newsgroups(categories=categories, subset='test', shuffle=True, random_state=108)

# Step 7
counter = CountVectorizer()

# Step 8
counter.fit(test_emails.data + train_emails.data)

# Step 9
train_counts = counter.transform(train_emails.data)

# Step 10
test_counts = counter.transform(test_emails.data)

# Step 11
classifier = MultinomialNB()

# Step 12
classifier.fit(train_counts, train_emails.target)

# Step 13
print('The accuracy of the classifier on the initial dataset is:', classifier.score(test_counts, test_emails.target),
      "\n")

# Step 14 (Change the categories to be two more contrasting subjects to see if the classifier's accuracy rises)
categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey']
train_emails = fetch_20newsgroups(categories=categories, subset='train', shuffle=True, random_state=108)
test_emails = fetch_20newsgroups(categories=categories, subset='test', shuffle=True, random_state=108)

counter.fit(test_emails.data + train_emails.data)

train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier.fit(train_counts, train_emails.target)

print('The accuracy of the classifier on the second dataset is:', classifier.score(test_counts, test_emails.target))
