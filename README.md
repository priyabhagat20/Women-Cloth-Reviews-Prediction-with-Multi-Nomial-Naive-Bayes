# Women-Cloth-Reviews-Prediction-with-Multi-Nomial-Naive-Bayes
**Objective**

The objective is to classify reviews into discrete categories (ratings) using the Multinomial Naive Bayes classifier, which is particularly suited for text classification tasks.

**Importing Libraries**

The notebook begins by importing the necessary libraries:

* pandas for data manipulation.
* numpy for numerical operations.
* matplotlib.pyplot and seaborn for data visualization.
* sklearn for machine learning tasks.

**Data Import and Initial Inspection**

The dataset, Women Clothing E-Commerce Review.csv, is imported using pandas.read_csv(). The notebook then inspects the data using head(), info(), and shape to understand its structure, the types of data it contains, and the number of rows and columns.

**Handling Missing Values**

The notebook identifies and handles missing values in the 'Review' column:

* It replaces empty strings with NaN.
* It then fills NaN values with the string "No Review".

**Defining Target and Features**

The target variable (y) is set as the 'Rating' column, which contains the ratings given by users. The feature variable (X) is set as the 'Review' column, which contains the textual reviews.

**Splitting the Data**

The dataset is split into training and testing sets using train_test_split from sklearn.model_selection, with 70% of the data used for training and 30% for testing. The split is stratified to ensure that each subset has the same distribution of ratings as the original dataset.

**Text Conversion to Tokens**

Text data in the reviews is converted into numerical features using CountVectorizer from sklearn.feature_extraction.text. The vectorizer transforms the text into a matrix of token counts, considering bigrams and trigrams, and removing English stop words. The number of features is limited to 5000.

**Model Training**

A Multinomial Naive Bayes model from sklearn.naive_bayes is instantiated and trained on the tokenized training data using model.fit().

**Model Prediction**

Predictions are made on the test set using model.predict(). The shape and content of the predictions are displayed.

**Model Evaluation**

The model's performance is evaluated using:

* Confusion matrix: to see the distribution of true vs. predicted classes.
* Classification report: to get precision, recall, f1-score, and support for each class.

**Recategorizing Ratings**

The ratings are recategorized into binary classes:

* Ratings 1, 2, and 3 are grouped as 0 (Poor).
* Ratings 4 and 5 are grouped as 1 (Good).

**Re-splitting the Data**

The data is re-split into training and testing sets following the same procedure as before, but with the new binary target variable.

**Re-training the Model**

The Multinomial Naive Bayes model is re-trained on the new binary target variable.

**Re-evaluating the Model**

The model is evaluated again using the same metrics (confusion matrix and classification report) to assess its performance on the binary classification task.
