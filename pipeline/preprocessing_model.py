'''
This script contains the necessary steps to import and process Steam Store data
in a format matching that of the dataset found on Kaggle.
It cleans the imported CSV file, saves a copy as a new CSV, and returns
a pandas dataframe ready to be used in model building.

The dataframe is then used to build a model and fit it to a training set.
The data is split into training, validation, and test sets. It is shuffled
and prepared for a (to be edited) model.
The function returns a model object as well as the accuracies between
the training and validation sets to check performance and overfitting.
'''
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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


stop_words = set(stopwords.words('english'))

def load_and_prepare_data(file_path):
    '''
    Load the dataset and prepare it for processing.

    Parameters:
    file_path (str): The path to the dataset file.

    Returns:
    pd.DataFrame: The loaded dataset.
    '''
    df = pd.read_csv(file_path)
    return df

def convert_release_date(df):
    '''
    Convert release_date to datetime and calculate age_of_game.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the release_date column.

    Returns:
    pd.DataFrame: The DataFrame with the age_of_game column added.
    '''
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d %b, %Y', errors='coerce')
    current_date = datetime.now()
    df['age_of_game'] = df['release_date'].apply(lambda x: current_date.year - x.year if pd.notnull(x) else None)
    return df

def clean_discounted_price(df):
    '''
    Clean the discounted_price column by handling 'Free' values and removing currency symbols.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the discounted_price column.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    '''
    df['discounted_price'] = df['discounted_price'].replace('Free', '0')
    df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['discounted_price'] = np.log1p(df['discounted_price'])
    return df

def remove_invalid_release_dates(df, cutoff_date='2024-05-09'):
    '''
    Remove games with release dates after the data was scraped.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the release_date column.
    cutoff_date (str): The cutoff date to filter the release_date column.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    '''
    cutoff = pd.to_datetime(cutoff_date)
    df = df[df['release_date'] <= cutoff]
    return df

def split_genres(df, column='genres'):
    '''
    One-hot encode the genres column.

    Parameters:
    df (pd.DataFrame): The DataFrame with a genres column.
    column (str): The name of the genres column to be split.

    Returns:
    pd.DataFrame: The DataFrame with one-hot encoded genre columns.
    '''
    genres_split = df[column].str.get_dummies(sep=', ')
    df = pd.concat([df, genres_split], axis=1)
    return df

def transform_to_binary_class(df, column_name, threshold):
    '''
    Transform a numerical column into binary classes.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to transform.
    column_name (str): The name of the column to transform.
    threshold (float): The threshold to determine binary classes.

    Returns:
    pd.DataFrame: The DataFrame with a new binary class column.
    '''
    df['binary_class'] = df[column_name].apply(lambda x: 'positive' if x >= threshold else 'negative')
    return df

def preprocess_text(text, stop_words):
    '''
    Preprocess text by lowercasing, removing punctuation, and stopwords.

    Parameters:
    text (str or other): The text to preprocess. If not a string, it will be converted to a string.
    stop_words (set): A set of stopwords to remove from the text.

    Returns:
    str: The preprocessed text.
    '''
    # Ensure the input is a string
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def preprocess_dataframe(df, text_column):
    '''
    Apply text preprocessing to a specified column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text column.
    text_column (str): The name of the text column to preprocess.

    Returns:
    pd.DataFrame: The DataFrame with preprocessed text.
    '''
    df[text_column] = df[text_column].apply(lambda x: preprocess_text(x, stop_words))
    return df

def tokenize_text(df, text_column, num_words=5000, oov_token='<OOV>'):
    '''
    Tokenize the text data and limit the vocabulary size.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text column to tokenize.
    text_column (str): The name of the text column.
    num_words (int): The maximum number of words to keep in the tokenizer's vocabulary.
    oov_token (str): The token for out-of-vocabulary words.

    Returns:
    pd.DataFrame: The DataFrame with tokenized text.
    Tokenizer: The fitted Keras tokenizer instance.
    '''
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(df[text_column])
    df['tokenized_text'] = tokenizer.texts_to_sequences(df[text_column])
    return df, tokenizer

def truncate_pad_data(sequences, max_length):
    '''
    Pads and truncates sequences to a specified maximum length.

    Parameters:
    sequences (list of list of int): The tokenized text sequences to pad/truncate.
    max_length (int): The maximum length for the sequences.

    Returns:
    np.ndarray: The padded/truncated sequences as a NumPy array.
    '''
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post', truncating='post', value=0)
    return padded_data

def preprocess_numerical_features(train_df, test_df, columns):
    '''
    Scale numerical features using StandardScaler and replace the original columns with the scaled values.

    Parameters:
    train_df (pd.DataFrame): The training dataset.
    test_df (pd.DataFrame): The test dataset.
    columns (list): The list of columns to scale.

    Returns:
    pd.DataFrame, pd.DataFrame: The training and test datasets with scaled numerical features.
    '''
    scaler = StandardScaler()
    train_df[columns] = scaler.fit_transform(train_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])

    return train_df, test_df

def encode_labels(df, column):
    '''
    Encode binary class labels.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the binary class column.
    column (str): The name of the binary class column.

    Returns:
    pd.Series: The encoded labels.
    '''
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(df[column])

def process_datasets(X_train, X_val, X_test, text_column, max_length=35, num_words=5000):
    '''
    Preprocess, tokenize, and pad text data in training, validation, and test datasets.

    Parameters:
    X_train (pd.DataFrame): The training dataset.
    X_val (pd.DataFrame): The validation dataset.
    X_test (pd.DataFrame): The test dataset.
    text_column (str): The name of the text column to process.
    max_length (int): The maximum length for padding sequences.
    num_words (int): The maximum number of words to keep in the tokenizer's vocabulary.

    Returns:
    tuple: The processed training, validation, and test datasets.
    '''
    # Preprocess text
    X_train = preprocess_dataframe(X_train, text_column)
    X_val = preprocess_dataframe(X_val, text_column)
    X_test = preprocess_dataframe(X_test, text_column)

    # Tokenize text
    X_train, tokenizer = tokenize_text(X_train, text_column, num_words=num_words)
    X_val['tokenized_text'] = tokenizer.texts_to_sequences(X_val[text_column])
    X_test['tokenized_text'] = tokenizer.texts_to_sequences(X_test[text_column])

    # Pad sequences
    X_train['padded_text'] = list(truncate_pad_data(X_train['tokenized_text'], max_length=max_length))
    X_val['padded_text'] = list(truncate_pad_data(X_val['tokenized_text'], max_length=max_length))
    X_test['padded_text'] = list(truncate_pad_data(X_test['tokenized_text'], max_length=max_length))

    # Drop unnecessary columns
    X_train.drop(columns=[text_column, 'tokenized_text'], inplace=True)
    X_val.drop(columns=[text_column, 'tokenized_text'], inplace=True)
    X_test.drop(columns=[text_column, 'tokenized_text'], inplace=True)

    return X_train, X_val, X_test


def data_cleaning(df):
    '''
    Main function to clean and preprocess the data.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean and preprocess.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    '''
    df = convert_release_date(df)
    df = clean_discounted_price(df)
    df = remove_invalid_release_dates(df)
    df = split_genres(df)

    return df

def data_processing(df):
    '''
    Main function to process the data, including splitting, encoding, and feature preparation.

    Parameters:
    df (pd.DataFrame): The cleaned DataFrame.

    Returns:
    tuple: The processed training, validation, and test datasets along with labels.
    '''
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

    # Split data into train and test
    train_df, test_df = train_test_split(subset, test_size=0.2, random_state=42)

    # Transform review percentage to binary classes
    threshold = train_df['overall_review_%'].median()
    train_df = transform_to_binary_class(train_df, 'overall_review_%', threshold)
    test_df = transform_to_binary_class(test_df, 'overall_review_%', threshold)

    # Drop unnecessary columns
    columns_to_drop = ['overall_review_%']
    train_df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)

    # Preprocess numerical features (replacing the original columns)
    numerical_columns = ['discounted_price', 'age_of_game', 'awards']
    train_df, test_df = preprocess_numerical_features(train_df, test_df, numerical_columns)

    # Create validation set from the training data
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

    # Preprocess text features and drop unnecessary columns
    X_train, X_val, X_test = process_datasets(train_df, val_df, test_df, 'about_description')

    # Encode labels
    y_train = encode_labels(train_df, 'binary_class')
    y_val = encode_labels(val_df, 'binary_class')
    y_test = encode_labels(test_df, 'binary_class')

    # Drop target from X sets
    columns_to_drop = ['binary_class']
    X_train.drop(columns=columns_to_drop, inplace=True)
    X_val.drop(columns=columns_to_drop, inplace=True)
    X_test.drop(columns=columns_to_drop, inplace=True)


    return X_train, X_val, X_test, y_train, y_val, y_test
    
def prepare_combined_features(X_train, X_val, X_test):
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
    X_train_padded = np.array(X_train['padded_text'].tolist())
    X_val_padded = np.array(X_val['padded_text'].tolist())
    X_test_padded = np.array(X_test['padded_text'].tolist())

    # Get all columns except 'padded_text'
    feature_columns = [col for col in X_train.columns if col != 'padded_text']

    # Combine padded text with other features
    X_train_features = X_train[feature_columns].to_numpy()
    X_val_features = X_val[feature_columns].to_numpy()
    X_test_features = X_test[feature_columns].to_numpy()

    X_train_combined = np.hstack((X_train_padded, X_train_features))
    X_val_combined = np.hstack((X_val_padded, X_val_features))
    X_test_combined = np.hstack((X_test_padded, X_test_features))

    return X_train_combined, X_val_combined, X_test_combined



# Final model setup with improved regularization and adjustments
def create_improved_model(vocab_size=5000, embedding_dim=2, num_other_features=10,
                          dense_units=32, dropout_rate=0.4, learning_rate=0.001, max_token_length = 35):
    '''Build an improved Keras model using the Sequential API with regularization.'''

    # Sequential model for text embeddings
    text_model = Sequential(name="TextModel")
    text_model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    text_model.add(layers.GlobalAveragePooling1D())

    # Sequential model for additional features
    features_model = Sequential(name="FeaturesModel")
    features_model.add(layers.Input(shape=(num_other_features,)))

    # Define a combined model using functional API to merge Sequential models
    combined_input = tf.keras.Input(shape=(max_token_length + num_other_features,), name='combined_input')

    # Process text part separately
    text_out = text_model(combined_input[:, :max_token_length])

    # Process features part separately
    features_out = combined_input[:, max_token_length:]

    # Concatenate the processed outputs
    concatenated = layers.concatenate([text_out, features_out], name='Concatenate')

    # Define dense layers with improved regularization
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(concatenated)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(int(dense_units / 2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    # Build final model
    model = models.Model(inputs=combined_input, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )
    return model


def accuracy(model, X, Y):
    '''
    function to print accuracy of model
    using score function from XGBoost
    model: model to evaluate
    X: features
    Y: labels
    '''
    loss, acc, auc = model.evaluate(X,Y,verbose=0)
    print("Accuracy: %.2f%%" % (acc * 100.0))
    print(f"AUC: {auc:.3f}")
    return loss, acc, auc

def main(path):
    '''
    main function for pipeline from
    input data (csv) to built and fitted model
    '''
    # Process data
    print(f'loading data from path {path}')
    df = load_and_prepare_data(path)
    print(f'performing data cleaning')
    df_cleaned = data_cleaning(df)
    print(f'splitting data')
    X_train, X_val, X_test, y_train, y_val, y_test = data_processing(df_cleaned)
    X_train_combined, X_val_combined, X_test_combined = prepare_combined_features(X_train, X_val, X_test)
    print(X_train_combined.shape)

    # Ensure reproducibility
    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    
    print(f'building model...')
    # Build the improved model
    model = create_improved_model(
        vocab_size=5000,
        embedding_dim=2,
        num_other_features=X_train.shape[1]-1,
        dense_units=32,
        dropout_rate=0.4,  # Increased dropout for regularization
        learning_rate=0.001
    )
    
    # Train the improved model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print(f'fitting model...')
    history = model.fit(
        x=X_train_combined,
        y=y_train,
        validation_data=(X_val_combined, y_val),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )

    print(f'creating plots...')
    # Plot loss for train and validation
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(history.history['loss'], lw=2, color='darkgoldenrod')
    plt.plot(history.history['val_loss'], lw=2, color='indianred')
    plt.legend(['Train', 'Validation'], fontsize=10)
    ax.set_xlabel('Epochs', size=10)
    ax.set_title('Loss')
    
    # Plot accuracy for train and validation
    ax = fig.add_subplot(1, 3, 2)
    plt.plot(history.history['binary_accuracy'], lw=2, color='darkgoldenrod')
    plt.plot(history.history['val_binary_accuracy'], lw=2, color='indianred')
    plt.legend(['Train', 'Validation'], fontsize=10)
    ax.set_xlabel('Epochs', size=10)
    ax.set_title('Accuracy')
    
    plt.show()
    fig.savefig('steam_nn_train_val_plots.pdf', format='pdf')

    model_name = 'steam_games_nn_model.pkl'
    joblib.dump(model, model_name)

    print(f'model saved! {model_name}')

    print('Training:')
    accuracy(model, X_train_combined, y_train)
    # eval_loss_auc(model, X_train, y_train)

    print('Validation:')
    accuracy(model, X_val_combined, y_val)
    # eval_loss_auc(model, X_val, y_val)

    print('Test:')
    accuracy(model, X_test_combined, y_test)
    # eval_loss_auc(model, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Steam Store data CSV file and build model")
    parser.add_argument("csv_path", help="Path to CSV data file")
    args = parser.parse_args()
    
    main(args.csv_path)
