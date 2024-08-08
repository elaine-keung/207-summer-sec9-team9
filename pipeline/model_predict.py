"""
This script takes an input set of data (which includes labels), processes it 
to match the format of the input of the model, and obtains a 
prediction using the model.
It returns the printed accuracy, log loss, and AUC score.
It creates a plot of the AUC and error and saves it as a pdf.
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
import preprocessing_model

def load_data(path):
    df = preprocessing_model.clean_data(path)
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
    desc = df['about_description']
    
    max_num_words = 5000  
    # length of sequence
    max_sequence_length = 150 
    # initialize and fit tokenizer
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(desc)
    X_seq = tokenizer.texts_to_sequences(desc)
    X_pad = pad_sequences(X_seq, maxlen=max_sequence_length, padding='post')
    tfidf_vectorizer = TfidfVectorizer(max_features=max_num_words)
    X_tfidf = tfidf_vectorizer.fit_transform(desc).toarray()
    X = np.hstack([X_tfidf, X_pad, X])
    
    return X, y
    


def main(path):
    model = joblib.load('steam_games_xgb_model.pkl')
    X, Y = load_data(path)

    preprocessing_model.accuracy(model, X, Y)

    log_loss, roc_auc = preprocessing_model.eval_loss_auc(model, X, Y)

    # Get evaluation results
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    
    # Plot training and validation AUC
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='AUC')
    ax.plot(x_axis, results['validation_0']['error'], label='Error')
    ax.legend()
    plt.ylabel('AUC/Error')
    plt.title('XGBoost AUC and Error')
    
    # Add test AUC to the plot
    plt.axhline(y=roc_auc, color='r', linestyle='--', label=f'AUC: {roc_auc:.2f}')
    plt.legend()
    plt.show()
    
    # Save the plot as a PDF
    fig.savefig('xgboost_auc_error_plot.pdf', format='pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make prediction with model and print result")
    parser.add_argument("csv_path", help="Path to CSV data file")
    args = parser.parse_args()
    
    main(args.csv_path)