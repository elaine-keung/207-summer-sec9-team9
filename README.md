# 207-summer-sec9-team9

## Unlocking Game Success: Predicting Game Reviews from Key Attributes?

### Data
For this project, we are using the (Steam Store Data from Kaggle)[https://www.kaggle.com/datasets/amanbarthwal/steam-store-data/data], scraped May 2024. The data can be downloaded as a CSV file from Kaggle and directly input into the EDA/processing notebook (or script file, as a path). The input to the modeling notebooks is the CSV which is saved as the output from the data processing.

### Repository Structure

The repository includes several directories and files:
- ```Final Presentation.pdf```, a .pdf of the final slide deck used in the presentation.
- ```data_preprocessing.ipynb```, an .ipynb notebook which includes code to clean the data (input being directly downloaded from Kaggle) and produce plots, as well as a final output cleaned CSV file.
- ```model.ipynb```, an .ipynb notebook which includes our final model, utilizing a neural network. The input is the CSV which was output from ```data_preprocessing.ipynb```. The code will split and shuffle the data, prepare the embeddings and train the model. Plots are produced to show the model's performance on the training and validation sets, with accuracy as the metric. The model is run on the test set and the metric and plots for this are also created as an output. The metrics are printed directly as outputs, and the plots are saved as .pdf files.
- ```extra_models/``` is a directory which includes notebooks for the other models that were attempted while looking for the best performing model.
- ```pipeline/``` is a directory which includes a ```___init___.py```, ```preprocessing_model.py```, and ```model_predict.py```. These scripts enable the code from ```data_preprocessing.ipynb``` and ```model.ipynb``` to be run from the command line. ```preprocessing_model.py``` takes one argument which is the path to a dataset (CSV). It runs the data cleaning steps as well as the model building steps, including shuffling/splitting. This script returns a ```.pkl``` file of the trained model, and saves a copy of the cleaned dataset before it has been split for model training. ```model_predict.py``` takes one argument, a path to a dataset (CSV), and imports the trained model to create predictions on that dataset. It cleans the data with the same steps to match the format of data which the model can take as an input, then creates predictions and returns them. If the imported dataset includes a way to validate the results, such as an included label column, it will calculate and print the accuracy. It will also return the performance plot as a saved .pdf image.

### Contributions 
| Member    | Approx. Hours | Tasks |
| -------- | ------- |------- |
| Ananya  | -------  |------- |
| Elaine | -------   |------- |
| Mia    | -------  |------- |
| Francisco  | -------  |------- |
