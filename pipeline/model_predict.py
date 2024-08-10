"""
This script takes an input set of data (which includes labels), processes the entire set 
to match the format of the input of the model, and obtains a 
prediction using the model.
It returns the printed accuracy, log loss, and AUC score.
It creates a plot of the AUC and error and saves it as a pdf.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import os
import random
import re
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, GlobalAveragePooling1D, Dense
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
import preprocessing_model
warnings.filterwarnings('ignore')

def process_datasets_abbr(df, text_column, max_length=35, num_words=5000):
    '''
    Preprocess, tokenize, and pad text data in training, validation, and test datasets.

    Parameters:
    df (pd.DataFrame): dataset
    text_column (str): The name of the text column to process.
    max_length (int): The maximum length for padding sequences.
    num_words (int): The maximum number of words to keep in the tokenizer's vocabulary.

    Returns:
    df: The processed df.
    '''
    # Preprocess text
    df = preprocessing_model.preprocess_dataframe(df, text_column)

    df, tokenizer = preprocessing_model.tokenize_text(df, text_column, num_words=num_words)
    # Pad sequences
    df['padded_text'] = list(preprocessing_model.truncate_pad_data(df['tokenized_text'], max_length=max_length))

    # Drop unnecessary columns
    df.drop(columns=[text_column, 'tokenized_text'], inplace=True)

    return df

def prepare_combined_abbr(df):
    '''
    Prepare combined features by converting padded text to NumPy arrays and combining them with all other features.

    Parameters:
    X_train (pd.DataFrame): The training dataset.
    X_val (pd.DataFrame): The validation dataset.
    X_test (pd.DataFrame): The test dataset.

    Returns:
    tuple: Combined feature arrays for training, validation, and test datasets.
    '''
    # Convert padded text to NumPy arrays
    df_padded = np.array(df['padded_text'].tolist())
    # Get all columns except 'padded_text'
    feature_columns = [col for col in df.columns if col != 'padded_text']
    # Combine padded text with other features
    df_features = df[feature_columns].to_numpy()
    df_combined = np.hstack((df_padded, df_features))
   
    return df_combined
    
def data_processing_abbr(df):
    '''
    Abbreviated function to process the data, including splitting, encoding, and feature preparation.

    Parameters:
    df (pd.DataFrame): The cleaned DataFrame.

    Returns: df with created binary label, prepared for model prediction
    '''

    df = preprocessing_model.data_cleaning(df)
    # Drop unnecessary columns
    subset = df[['discounted_price', 'about_description', 'awards', 'overall_review_%',
       'age_of_game', 'Action', 'Adventure', 'Animation & Modeling',
       'Audio Production', 'Casual', 'Design & Illustration', 'Early Access',
       'Education', 'Free to Play', 'Game Development', 'Indie',
       'Massively Multiplayer', 'Movie', 'RPG', 'Racing', 'Simulation',
       'Software Training', 'Sports', 'Strategy', 'Utilities',
       'Video Production', 'Web Publishing']]
    # Drop any rows with null values before any processing
    subset.dropna(inplace=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f'steam_games_data_cleaned_{timestamp}.csv'
    print(f'Cleaning finished; saving copy to {filename}')
    subset.to_csv(filename)
    # Transform review percentage to binary classes
    subset['binary_class'] = subset['overall_review_%'].apply(lambda x: 1 if x > subset['overall_review_%'].median() else 0)

    # Drop unnecessary columns
    subset.drop(columns={'overall_review_%'}, inplace=True)

    # Preprocess numerical features (replacing the original columns)
    scaler = StandardScaler()
    subset[['discounted_price', 'age_of_game', 'awards']] = scaler.fit_transform(subset[['discounted_price', 'age_of_game', 'awards']])

    subset = process_datasets_abbr(subset, 'about_description')
    
    return subset

def load_data(path):
    '''
    Reads data from a path and cleans it.
    Only splits the features from the labels.
    Returns features, labels
    '''
    print(f'loading data from path {path}')
    df = preprocessing_model.load_and_prepare_data(path)
    df = data_processing_abbr(df)
    X = df.drop(columns='binary_class')
    Y = df[['binary_class']]
    X = prepare_combined_abbr(X)
    # Encode labels
    Y = preprocessing_model.encode_labels(Y, 'binary_class')
    
    return X, Y
    
def save_report(Y, pred_class, pred, pdf_path):
    '''
    Creates a pdf report including a confusion matrix, classification metrics, 
    ROC Curve plot, and precision/recall curve.

    Input:
    Y: true labels
    pred_class: probability for positive class
    pred: predicted labels
    pdf_path: path to save pdf report
    
    '''
    with PdfPages(pdf_path) as pdf:
        # Confusion Matrix
        conf_matrix = confusion_matrix(Y, pred_class)
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['0', '1'])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        pdf.savefig(fig)
        plt.close(fig)

        # ROC Curve
        fpr, tpr, _ = roc_curve(Y, pred)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        pdf.savefig(fig)
        plt.close(fig)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(Y, pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        pdf.savefig(fig)
        plt.close(fig)

def main(path):
    '''
    main function for pipeline
    loads in data from path and loads in pre-trained model
    from preprocessing_model.py
    returns pdf report containing plots and dataframe CSV 
    of original labels vs predicted labels
    '''
    print('Loading model and data...')
    model = joblib.load('steam_games_nn_model.pkl')
    X, Y = load_data(path)

    print('Predicting...')
    pred = model.predict(X)
    pred_df = pd.DataFrame(pred, columns=['prediction'])
    pred_class = (pred > 0.5).astype(int)
    loss, acc, auc = preprocessing_model.accuracy(model, X, Y)

    # Get evaluation results
    print('Organizing results...')
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f'steam_games_model_predictions_report_{timestamp}.pdf'
    print(f'Predictions finished; saving report to {filename}')
    save_report(Y, pred_class, pred, f'{filename}')

    df_pred = pd.concat([pd.DataFrame(Y, columns=['original_class']), pred_df], axis=1)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f'steam_games_model_predictions_{timestamp}.csv'
    print(f'Saving dataframe copy to {filename}')
    df_pred.to_csv(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make prediction with model and print result")
    parser.add_argument("csv_path", help="Path to CSV data file")
    args = parser.parse_args()
    
    main(args.csv_path)