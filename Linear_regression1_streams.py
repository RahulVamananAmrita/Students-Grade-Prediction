#step1:importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import streamlit
#step2:Imports the dataset into the program
#      by the help of “Pandas” library.

dataset = pd.read_csv('student_scores.csv')

#So in our problem attribute=”Hours”, Labels=”Score”

X = dataset.iloc[:, :-1].values    #Attributes,Hours(indepedent) 
y = dataset.iloc[:, 1].values      # Labels , Score (Depend)

#Split this data into training and testing data.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Train the algorithm .

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

#y_pred = regressor.predict(X_test)


def lr_prediction(var_1):
    #   X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    #   lr.fit(X_train,y_train)
       array=np.array([var_1])
       a=array.reshape(-1, 1)
       model_prediction=regressor.predict(a)
     #  array.reshape(-1, 1)
       return model_prediction


def run():
     streamlit.title("Students Score prediction Model")
     html_temp="""
     """
     streamlit.markdown(html_temp)
     Hours=streamlit.text_input("Hours Studied by student")
     prediction=""
     if streamlit.button("Predict"):
          prediction=lr_prediction(Hours)

     streamlit.success("Student Can Score Mark in Percentage(%):{}".format(prediction))

if __name__=='__main__':
     run()
#streamlit(run app.py)
