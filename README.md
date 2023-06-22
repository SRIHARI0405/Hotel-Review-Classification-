# Hotel_Review_Classification
This is a NLP project which predicts the reviews given by a customer is positive or negative to the services provided by the hotel
The objective of the analysis is to predict that the customer has been satisfied or not and the rating based on review. Model gives Positive if the expected rating is >=3 out of 5 or else Negative.
The data here is text so preprocessing like removing punctuations, numbers, extra white spaces is done before performing EDA.
Analyzed the data with several EDA methods and tf-idf vectorization of text is done to perform the model building
Several classification techniques such as Gaussian Naive Bayes, Decision Tree, Random Forest, Logistic Regression, Adaboost, K-Nearest Neighbours are used to classify the ratings based on the review and finalized logistic regression model i.e having train accuracy of 94.7% and test accuracy of 92.6%
Finally The Model is deployed using Streamlit

