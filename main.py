import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import warnings
import pickle
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.preprocessing import LabelEncoder


class Regression_model :
    try :
        def __init__(self):
            self.df = pd.read_csv("data.csv")

    except Exception as e:
        error_type, error_msg, exc_tb = sys.exc_info()
        print(f"Error: {error_msg}, line: {exc_tb.tb_lineno}")

    # here cleaning the data and remove the unwanted features
    def data_cleaning(self):
        try:
            self.df = self.df.drop(['date', 'street', "statezip"], axis=1)
            le = LabelEncoder()
            self.df["city"] = le.fit_transform(self.df["city"])
            self.df["country"] = self.df["country"].map({"USA": 1})
        except Exception as e:
            error_type, error_msg, exc_tb = sys.exc_info()
            print(f'error type was {error_type},due to {error_msg},in the line of {exc_tb.tb_lineno}')


    # split the data independent and dependent data
    def data_split(self):
        try:
            self.X_data= self.df.iloc[:, 1:]
            self.y_data = self.df.iloc[:, 0]
        except Exception as e:
            error_type, error_msg, exc_tb = sys.exc_info()
            print(f'error type was {error_type},due to {error_msg},in the line of {exc_tb.tb_lineno}')

    # split train and test data
    def split_train_test_data(self):
        try:
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X_data,self.y_data,test_size=0.2,random_state=42)
        except Exception as e:
            error_type, error_msg, exc_tb = sys.exc_info()
            print(f'error type was {error_type},due to {error_msg},in the line of {exc_tb.tb_lineno}')

    # train algorithm  using train data
    def model_training(self):
        try:
            self.reg = LinearRegression()  # create the obj for class
            self.reg.fit(self.X_train,self.y_train)
        except Exception as e:
            error_type, error_msg, exc_tb = sys.exc_info()
            print(f'error type was {error_type},due to {error_msg},in the line of {exc_tb.tb_lineno}')

    # find the train_performance
    def train_performance(self):
        try:
            self.train_predit_point = self.reg.predict(self.X_train)
            self.train_r2_score = r2_score(self.y_train,self.train_predit_point)
            self.train_rmse = root_mean_squared_error(self.y_train,self.train_predit_point)
            print(f'train r2_score {self.train_r2_score},train rmse{self.train_rmse}')


        except Exception as e:
            error_type, error_msg, exc_tb = sys.exc_info()
            print(f'error type was {error_type},due to {error_msg},in the line of {exc_tb.tb_lineno}')

    # find the test performance
    def test_performance(self):
        try:
            self.test_predit_point = self.reg.predict(self.X_test)
            self.test_r2_score = r2_score(self.y_test, self.test_predit_point)
            self.test_rmse = root_mean_squared_error(self.y_test, self.test_predit_point)
            print(self.test_rmse,self.test_r2_score)

        except Exception as e:
            error_type, error_msg, exc_tb = sys.exc_info()
            print(f'error type was {error_type},due to {error_msg},in the line of {exc_tb.tb_lineno}')


    # store in a pickle file
    def pickle(self):
        try:  # save the model
            with open('houseprice.pkl', 'wb') as a:
                pickle.dump(self.reg, a)
            print('model save successfully')
            # load the model
            with open('houseprice.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print('model loaded successfully')

        except Exception as e:
            error_type, error_msg, exc_tb = sys.exc_info()
            print(f'error type was {error_type},due to {error_msg},in the line of {exc_tb.tb_lineno}')

    # using this input check weather the predicted or not
    def user_input(self):
        try:
            bedrooms = int(input('enter no of bedrooms:'))
            bathrooms = float(input('enter no of bathrooms'))
            sqft_living = float(input("Enter living area (sqft): "))
            sqft_lot = float(input("Enter lot area (sqft): "))
            floors = float(input("Enter number of floors: "))
            waterfront = int(input("Enter waterfront: "))
            view = int(input("Enter view rating: "))
            condition = int(input("Enter condition: "))
            sqft_above = float(input("Enter sqft above: "))
            sqft_basement = float(input("Enter sqft basement: "))
            yr_built = int(input("Enter year built: "))
            yr_renovated = int(input("Enter year renovated:"))
            city = int(input("Enter city code: "))
            country = int(input("Enter country code: "))

            user_data = pd.DataFrame({'bedrooms': [bedrooms], 'bathrooms': [bathrooms],
                                  'sqft_living': [sqft_living], 'sqft_lot': [sqft_lot],
                                  'floors': [floors], 'waterfront': [waterfront],
                                  'view': [view], 'condition': [condition],
                                  'sqft_above': [sqft_above], 'sqft_basement': [sqft_basement],
                                  'yr_built': [yr_built], 'yr_renovated': [yr_renovated],
                                  'city': [city], 'country': [country] })
            print(self.model.predict(user_data))

        except Exception as e:
            error_type, error_msg, exc_tb = sys.exc_info()
            print(f'error type was {error_type},due to {error_msg},in the line of {exc_tb.tb_lineno}')


if __name__ == '__main__':
    try :
        obj = Regression_model()
        obj.data_cleaning()
        obj.data_split()
        obj.split_train_test_data()
        obj.model_training()
        obj.test_performance()
        obj.train_performance()
        obj.pickle()
        obj.user_input()
    except Exception as e:
        error_type, error_msg, exc_tb = sys.exc_info()
        print(f'error type was {error_type},due to {error_msg},in the line of {exc_tb.tb_lineno}')



