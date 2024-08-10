# Pipeline README.md

To use this pipeline, the dataset must be saved locally.

1. Train the model:

```python preprocessing_model.py path/to/file.csv```

This will return:
- Printed accuracies and AUC for train, validation, and test sets.
- A CSV file of the cleaned dataset, named in format ```steam_games_data_cleaned_[date]``` (saved in same directory). 
- ```steam_games_nn_model.pkl``` (saved in same directory), which will be used by the prediction script.

2. Predictions:

Predictions are done with another dataset, which must have the same columns as the original (e.g. more recently scraped data). The script will run the entire dataset through the model for predictions, it does not split the data. Ensure that ```steam_games_nn_model.pkl``` is located in the same directory.

```python model_predict.py path/to/file.csv```

This will return:
- A CSV file of the cleaned dataset, named in format ```steam_games_predictions_data_cleaned_[date]``` (saved in same directory)
- A PDF file containing a confusion matrix, ROC plot, and precision-recall plot.
- A CSV file of the original labels and predictions, named in format ```steam_games_model_predictions_[date].csv```
