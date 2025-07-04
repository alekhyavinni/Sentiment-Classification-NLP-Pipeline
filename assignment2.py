"""The project develops a sentiment analysis model to classify customer reviews."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# Load data
file_path = 'reviews.csv'
data = pd.read_csv(file_path, delimiter='\t')


def bin_ratings(df):
    """Categorizes reviews into sentiment values based on their rating."""
    df['Sentiment'] = df['RatingValue'].apply(
        lambda x: 0 if x in [1, 2] else (1 if x == 3 else 2)
    )
    return df


data = bin_ratings(data)


data['Number'] = range(1, len(data) + 1)

# Select only the relevant columns: Number, Sentiment, and Review
final_data = data[['Number', 'Sentiment', 'Review']]

print(final_data.head())

# Step 2: Balancing the Data
# Count the occurrences of each sentiment category
count_pos = len(final_data[final_data['Sentiment'] == 2])
count_neg = len(final_data[final_data['Sentiment'] == 0])
count_neutral = len(final_data[final_data['Sentiment'] == 1])

count_pos, count_neutral, count_neg

# Drop excess positive reviews to balance the dataset
data_positive = final_data[final_data['Sentiment'] == 2].sample(
    n=min(count_neg, count_neutral), random_state=42
)
data_negative = final_data[final_data['Sentiment'] == 0]
data_neutral = final_data[final_data['Sentiment'] == 1]

print(f"postive_reviews:{len(data_positive)}, negative_reviews:{len(data_negative)},Nuetral_reviews: {len(data_neutral)}")

balanced_data = pd.concat([data_positive, data_negative, data_neutral])

# Step 3: Splitting the Data into Training and Validation Sets
train_data, valid_data = train_test_split(balanced_data, test_size=0.2, random_state=42)

train_data['Number'] = range(1, len(train_data) + 1)
valid_data['Number'] = range(1, len(valid_data) + 1)

# Save the train and validation sets
train_data.to_csv('train.csv', index=False)
valid_data.to_csv('valid.csv', index=False)

# Step 4: Model Training and Evaluation
vectorizer = CountVectorizer(stop_words='english')
x_train = vectorizer.fit_transform(train_data['Review'])
y_train = train_data['Sentiment']

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)


# Define the evaluation function
def evaluate(file_name):
    """The function assesses the model's performance."""
    data = pd.read_csv(file_name)
    x_valid = vectorizer.transform(data['Review'])
    y_valid = data['Sentiment']

    y_pred = model.predict(x_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='macro')
    class_f1 = f1_score(y_valid, y_pred, average=None)
    conf_matrix = confusion_matrix(y_valid, y_pred, normalize='true')

    # Print the performance metrics
    print(f"Accuracy: {accuracy}")
    print(f"Average F1 Score (Macro): {f1}")
    print(f"Class-wise F1 Scores:")
    print(f"Negative: {class_f1[0]}")
    print(f"Neutral: {class_f1[1]}")
    print(f"Positive: {class_f1[2]}")
    print("Confusion Matrix (Normalized):")
    print(conf_matrix)


# Call the evaluation function on the validation data
evaluate('valid.csv')
