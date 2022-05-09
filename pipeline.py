import sklearn
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor

__authors__ = ['Adney Costa','Matheus Oliveira','Nívea de Abreu','Ricardo Mourão']


class Pipeline:
    def __init__(self,df) -> None:
        self.df = df
        self.transformer=preprocessing.OneHotEncoder()
    
    def split_data(self) -> None:
        self.X_train = self.df[self.df['year']<2016]
        self.X_test = self.df[(self.df['year']>2015) & (self.df['year']<2018)]
        self.y_train = self.X_train['destinated_area']
        self.y_test = self.X_test['destinated_area']
        self.X_train = self.X_train.drop('destinated_area',axis=1)
        self.X_test = self.X_test.drop('destinated_area',axis=1)
        self.X_forecast = self.df[self.df['year']>2017]
        self.y_forecast = self.X_forecast['destinated_area']
        self.X_forecast = self.X_forecast.drop('destinated_area',axis=1)

    def split_data_without_null(self)-> None:
        self.X_train = self.df[(self.df['year']<2016) & (self.df['year']<1985)]
        self.X_test = self.df[(self.df['year']>2015) & (self.df['year']<2018)]
        self.y_train = self.X_train['destinated_area']
        self.y_test = self.X_test['destinated_area']
        self.X_train = self.X_train.drop('destinated_area',axis=1)
        self.X_test = self.X_test.drop('destinated_area',axis=1)
        self.X_forecast = self.df[self.df['year']>2017]
        self.y_forecast = self.X_forecast['destinated_area']
        self.X_forecast = self.X_forecast.drop('destinated_area',axis=1)

    def encoder(self) -> None:
        self.X_train =  self.transformer.fit_transform(self.X_train)
        self.X_test =  self.transformer.fit_transform(self.X_test)
        self.X_forecast =  self.transformer.fit_transform(self.X_forecast)
    
    def decoder(self) -> None:
        self.X_train =  self.transformer.inverse_transform(self.X_train)
        self.X_test =  self.transformer.inverse_transform(self.X_test)
        self.X_forecast =  self.transformer.inverse_transform(self.X_forecast)

    def train_knnreg(self) -> None:
        """Testing the complexity of the params in KNeighborsRegressor model"""
        self.n_neighbors = 0 
        self.knn_score = 0
        for i in range(1,30):
            self.knnreg=KNeighborsRegressor(n_neighbors=i).fit(self.X_train,self.y_train)
            print(f'For this number of neighbors {i}, this is the score of the model: {self.knnreg.score(self.X_test,self.y_test)}')
            if self.knnreg.score(self.X_test,self.y_test)>self.knn_score:
                self.n_neighbors = i
        print(f"The best number of Neighbors is this shallow comparision is: {self.n_neighbors}")

        self.knnreg=KNeighborsRegressor(n_neighbors=self.n_neighbors).fit(self.X_train,self.y_train)

    def train_linreg(self) -> None:
        self.linreg= LinearRegression().fit(self.X_train,self.y_train)
        print('R-squared score (training): {:.3f}'
            .format(self.linreg.score(self.X_train, self.y_train)))
        print('R-squared score (test): {:.3f}'
            .format(self.linreg.score(self.X_test, self.y_test)))

    

    

    

        
