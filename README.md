# IMDb Movie Review Sentiment Analysis
Group project for my Applied Machine Learning class. Project aims to build an accurate model that predicts movie review sentiment.

## COMS 4995 Applied Machine Learning: IMDB Sentiment Analysis Project
The main dataset for this project is IMDB Dataset.csv. It is too large to include in this repository, so can be downloaded here: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

In this zip file, there should be the final report, the README file that you are reading, and a folder “Code” containing all the code written for this project. In that folder are three subfolders that should all contain code:

1) Data Preprocessing and EDA
AML_Project_Deliverable2.ipynb: A Python notebook containing data cleaning process and tokenization as well as exploratory data analysis of our dataset.
wordcloud_and_barcharts.ipynb: A Python notebook containing additional exploratory data analysis done on the dataset.
2) Creating Embeddings
README_createEmbeddings.md: A README file containing instructions on how to create the embeddings from the processed data using create_embeddings.py.
create_embeddings.py: The Python file containing the code to create the four embeddings that we will test with each of our models.
3) Models:
BERT_Model.ipynb: Code for running the Pre-trained BERT Transformer model using the data generated from create_embeddings.py.
DNN_Model.ipynb: Code for finding the optimal Deep Neural Network model for each embedding type using the data generated from create_embeddings.py.
KNN_Model.ipynb: Code for finding the optimal KNN model for each embedding type using the data generated from create_embeddings.py.
Logistic_Regression_Model.py: Code for finding the optimal Logistic Regression model for each embedding type using the data generated from create_embeddings.py.
Random_Forest_Model.ipynb: Code for finding the optimal Random Forest model for each embedding type using the data generated from create_embeddings.py. Code for exploratory data analysis from AML_Project_Deliverable2.ipynb is also present.
Please do not hesitate to reach out if there are any questions.
