import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle

# Load data from CSV into DataFrame
df = pd.read_csv(r'C:\Users\PARUL\Desktop\badal dia\tnc.csv')

# Split data into features (X) and labels (y)
X = df['title']
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)
pickle.dump(model, open('tandc.pkl', 'wb'))
tandc_model=pickle.load(open('tandc.pkl', 'rb'))
pickle.dump(vectorizer, open('tandcvector.pkl', 'wb'))
tandc_vector_form=pickle.load(open('tandcvector.pkl', 'rb'))



# Model evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Example usage for new texts
new_texts = ["This subscription service has hidden fees."]
X_new = vectorizer.transform(new_texts)
prediction = model.predict(X_new)
print(prediction)
if prediction[0] == 1:
    print("The terms and conditions document contains certain words.")
else:
    print("The terms and conditions document does not contain certain words.")
accuracy = accuracy_score(y_test, y_pred)
print("TERMS AND CONDITION ACCURACY:", accuracy)