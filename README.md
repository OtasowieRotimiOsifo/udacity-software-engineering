# Project Summary
This project is a web app where an emergency worker can input a new message and get classification results in several categories. The app includes visualization for the dataset that was used in the training and validation of the model.

There are three main parts to the project: ETL, ML-Pipline and the web app.
## ETL
This part is responsible for data extrataction, cleaning/transformation and storing the data in an sqlite database file. The python file for ETL is in the data directory. This directory also includes example datasets for training. 

## ML-Pipeline
This part is responsible for training the model that is used to classify the messages. The python module builds a machine learning pipeline that uses natural language processing. The pipeline includes text tokenization and vectoriztion, pos tagging with starting verb and a multioutput classifier. Grid search CV is used to obtain the best paramers for the model that does the classification. The trained model is stored in a pickle file for deployment. The python module is in the models directory; it uses the data that was stored in the database in the ETL phase for training. 

## Deployed App
The web app uses the model that was trained in the ML-Pipeline to perform message classification. It is also displays visualiztions for the training and validation dataset to the user. The trained model is loaded from a pickle file.

# How to run the different different modules
The following steps can be used to execute each of the modules from the command line while at the src directory.
## ETL
python ./data/process\_data.py ./data/disaster\_messages.csv ./data/disaster\_categories.csv ./DisasterResponse.db

## ML-Pipeline
python ./models/train\_classifier.py ./DisasterResponse.db ./classifier.pkl

## App
To run the app, do the followings:
python ./app/run.py ./DisasterResponse.db ./classifier.pkl

# Extra
It could be neccessary to install some dependent python packages before executing the modules. Check the imports in the files for details. An example for such a package is plotly.

I would recommend python 3.7 and above but is it not compulsory.
