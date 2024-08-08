"""
This script contains the necessary steps to import and process Steam Store data
in a format matching that of the dataset found on Kaggle.
It cleans the imported CSV file, saves a copy as a new CSV, and returns
a pandas dataframe ready to be used in model building.

The dataframe is then used to build a model and fit it to a training set.
The data is split into training, validation, and test sets. It is shuffled
and prepared for a (to be edited) model.
The function returns a model object as well as the accuracies between
the training and validation sets to check performance and overfitting.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, roc_auc_score
import joblib


def clean_data(file):
    '''
    steps to process data 
    format of file is expected to match kaggle publicized data
    returns a pandas dataframe intended to be used for model building
    exports a csv of dataframe
    '''
    
    # read file
    df = pd.read_csv(file)

    # convert release date to date format
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d %b, %Y', errors='coerce')
    # calculate the age of the game
    current_date = datetime.now()
    df['age_of_game'] = df['release_date'].apply(lambda x: current_date.year - x.year if pd.notnull(x) else None)
    # filter the dataframe for games with release dates after the cutoff date
    df[df['release_date'] > pd.to_datetime('2024-05-09')]
    # remove rows
    df = df[df['release_date'] <= pd.to_datetime('2024-05-09')]
    
    # clean discounted_price column
    df['discounted_price'] = df['discounted_price'].replace('Free', '0')
    df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)

    # split the genres column and create one-hot encoding
    genres_split = df['genres'].str.get_dummies(sep=', ')
    # merge the one-hot encoded columns back with the original DataFrame
    df = pd.concat([df, genres_split], axis=1)

    # binary classification designation
    # based on median value of score percentage to account for imbalance
    df['binary_class'] = df['overall_review_%'].apply(lambda x: 'positive' if x > df['overall_review_%'].median() else 'negative')

    # drop features (many missing values)
    df = df.drop(columns = ['original_price', 'discount_percentage', 'content_descriptor', 
                            'recent_review','recent_review_%', 'recent_review_count', 
                            'app_id', 'title', 'dlc_available', 'age_rating',
                            'win_support', 'mac_support','linux_support', 
                            'release_date', 'categories', 'developer', 'publisher'])
    # drop empty
    df = df.dropna()

    # export to save a copy
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'steam_games_data_cleaned_{timestamp}.csv'
    
    print(f'Cleaning finished; saving copy to {filename}')
    df.to_csv(filename)

    return df

def process_data(df):
    '''
    This function prepares an input dataframe for 
    model building and fitting.
    The input dataframe is expected to have column names and format
    matching the output of function clean_data.
    '''  
 
    X = df[['discounted_price',
        'awards',
        'age_of_game',
        'Action',
        'Adventure',
        'Animation & Modeling',
        'Audio Production',
        'Casual',
        'Design & Illustration',
        'Early Access',
        'Education',
        'Free to Play',
        'Game Development',
        'Indie',
        'Massively Multiplayer',
        'Movie',
        'RPG',
        'Racing',
        'Simulation',
        'Software Training',
        'Sports',
        'Strategy',
        'Utilities',
        'Video Production',
        'Web Publishing']].reset_index(drop=True)
    y = df['binary_class'].map(lambda x: 1 if x == 'positive' else 0).reset_index(drop=True)
    
    # split training and test sets
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=1234)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1234)
    
    # shuffle
    np.random.seed(0)
    indices = np.arange(X_train.shape[0])
    shuffled_indices = np.random.permutation(indices)
    X_train = X_train.iloc[shuffled_indices]
    y_train = y_train.iloc[shuffled_indices]

    # embedding
    desc = df['about_description']
    
    # split training and validation sets
    X_train_text, X_val_test_text, X_train_num, X_test_val_num, y_train, y_val_test = train_test_split(
        desc, X, y, test_size=0.4, random_state=1234)
    
    X_val_text, X_test_text, X_val_num, X_test_num, y_val, y_test = train_test_split(X_val_test_text, X_test_val_num, y_val_test, test_size=0.5, random_state=1234)
    
    # vectorization using TF-IDF
    # tokenization and padding
    # unique tokens
    max_num_words = 5000  
    # length of sequence
    max_sequence_length = 150 
    
    # initialize and fit tokenizer
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(X_train_text)
    
    X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
    X_val_sequences = tokenizer.texts_to_sequences(X_val_text)
    X_test_sequences = tokenizer.texts_to_sequences(X_test_text)
    
    # pad sequences
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post')
    X_val_padded = pad_sequences(X_val_sequences, maxlen=max_sequence_length, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post')
    
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=max_num_words)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text).toarray()
    X_val_tfidf = tfidf_vectorizer.transform(X_val_text).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text).toarray()
    
    # add to features
    X_train = np.hstack([X_train_tfidf, X_train_padded, X_train_num])
    X_val = np.hstack([X_val_tfidf, X_val_padded, X_val_num])
    X_test = np.hstack([X_test_tfidf, X_test_padded, X_test_num])

    print('Data preprocessing split; ready for model')

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(X, Y, X_val, Y_val, learning_rate=0.1, n_estimators=100, max_depth=6, subsample=0.5, min_child_weight=1, 
                colsample_bytree=0.5, eval_metric=["auc","error"], early_stopping_rounds=10, verbose=True):
    '''
    create an xgboost classifier model and fit
    returns a model which can be used to predict
    X: features
    Y: labels
    X_val: feature validation
    Y_val: label validation
    learning_rate: default 0.1
    n_estimators: default 100
    max_depth: default 6
    subsample: default 0.5
    min_child_weight: default 1
    colsample_bytree: default 0.5
    eval_metric: default ["auc","error"]
    early_stopping_rounds: default 10
    verbose: default True
    '''
    xgb_classifier = XGBClassifier(objective='binary:logistic',
                                   eval_metric=eval_metric,
                                   early_stopping_rounds=early_stopping_rounds,
                                   learning_rate=learning_rate,
                                   n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   subsample=subsample,
                                   min_child_weight=min_child_weight,
                                   colsample_bytree=colsample_bytree)

    # fit the model
    print('Fitting model...')
    model = xgb_classifier.fit(X, Y, eval_set=[(X_val, Y_val)], verbose=verbose)
    return model

def accuracy(model, X, Y):
    '''
    function to print accuracy of model
    using score function from XGBoost
    model: model to evaluate
    X: features
    Y: labels
    '''
    acc = model.score(X,Y)
    print("Accuracy: %.2f%%" % (acc * 100.0))

def eval_loss_auc(model, X, Y):
    '''
    evaluate model using log loss and ROC AUC
    model: model to evaluate
    X: features
    Y: labels
    '''
    y_pred_proba = model.predict_proba(X)
    print(f'Log loss: {log_loss(Y, y_pred_proba):.3f}')
    roc_auc = roc_auc_score(Y, y_pred_proba[:,1])
    print(f'AUC: {roc_auc:.3f}')
    return log_loss(Y, y_pred_proba), roc_auc


def main(path):
    '''
    main function for pipeline from
    input data (csv) to built and fitted model
    '''
    # Process data
    df = clean_data(path)
    X_train, X_val, X_test, y_train, y_val, y_test = process_data(df)

    # Build and evaluate model
    model = build_model(X_train, y_train, X_val, y_val,
                        learning_rate=0.08,
                        n_estimators=120,
                        max_depth=6,
                        early_stopping_rounds=20,
                        verbose=121)

    joblib.dump(model, 'steam_games_xgb_model.pkl')

    print('Training:')
    accuracy(model, X_train, y_train)
    eval_loss_auc(model, X_train, y_train)

    print('Validation:')
    accuracy(model, X_val, y_val)
    eval_loss_auc(model, X_val, y_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Steam Store data CSV file and build model")
    parser.add_argument("csv_path", help="Path to CSV data file")
    args = parser.parse_args()
    
    main(args.csv_path)
