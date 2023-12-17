import pandas as pd
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


# Load the dataset
dataset_path = 'DatasetFods.csv'
data = pd.read_csv(dataset_path,encoding='ISO-8859-1')

# Data Preprocessing
# Handle missing values and '-' values
data.fillna({'category_list': 'Unknown', 'funding_total_usd': 0, 'funding_rounds': 0}, inplace=True)
data['funding_total_usd'] = data['funding_total_usd'].replace('-', 0)

# Encode the 'category_list' feature using label encoding
label_encoder = LabelEncoder()
data['category_list'] = label_encoder.fit_transform(data['category_list'])

# Split the dataset into features (X) and the target variable (y)
X = data[['category_list', 'funding_total_usd', 'funding_rounds']]
y = data['status']

# Create and train the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Create a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_category = request.form.get('category_list')
        user_funding_total = float(request.form.get('funding_total_usd'))
        user_funding_rounds = int(request.form.get('funding_rounds'))

        # Preprocess user input
        user_category_encoded = label_encoder.transform([user_category])

        # Make a prediction using the model
        prediction = model.predict([[user_category_encoded[0], user_funding_total, user_funding_rounds]])

        # Calculate the percentage of 'operating' startups in the dataset
        operating_percentage = (data['status'] == 'operating').sum() / len(data) * 100

        # Determine if the startup is likely to succeed or fail
        result = "Likely to Succeed" if prediction[0] == 'operating' else "Likely to Fail"

        return render_template('result.html', result=result, operating_percentage=operating_percentage)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
