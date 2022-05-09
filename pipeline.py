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
        self.transformer=preprocessing.OrdinalEncoder()
    
    def split_data(self) -> None:
        self.X_train = self.df[self.df['year']<2016]
        self.X_test = self.df[(self.df['year']>2015) & (self.df['year']<2018)]
        self.y_train = self.X_train['destinated_area']
        self.y_test = self.X_test['destinated_area']
        self.X_train = self.X_train.drop('destinated_area',axis=1)
        self.X_test = self.X_test.drop('destinated_area',axis=1)
        # self.X_forecast = self.df[self.df['year']>2017]
        # self.y_forecast = self.X_forecast['destinated_area']
        # self.X_forecast = self.X_forecast.drop('destinated_area',axis=1)

    def split_data_without_null(self)-> None:
        self.X_train = self.df[(self.df['year']<2016) & (self.df['year']<1985)]
        self.X_test = self.df[(self.df['year']>2015) & (self.df['year']<2018)]
        self.y_train = self.X_train['destinated_area']
        self.y_test = self.X_test['destinated_area']
        self.X_train = self.X_train.drop('destinated_area',axis=1)
        self.X_test = self.X_test.drop('destinated_area',axis=1)
        # self.X_forecast = self.df[self.df['year']>2017]
        # self.y_forecast = self.X_forecast['destinated_area']
        # self.X_forecast = self.X_forecast.drop('destinated_area',axis=1)

    def encoder(self) -> None:
        self.X_train =  self.transformer.fit_transform(self.X_train)
        self.X_test =  self.transformer.fit_transform(self.X_test)
        # self.X_forecast =  self.transformer.fit_transform(self.X_forecast)
    
    def decoder(self) -> None:
        self.X_train =  self.transformer.inverse_transform(self.X_train)
        self.X_test =  self.transformer.inverse_transform(self.X_test)
        # self.X_forecast =  self.transformer.inverse_transform(self.X_forecast)

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

    def train_ridge(self) -> None:
        alpha_values=[1e-3,0.1,1,2,5,10,50,100]
        self.best_alpha_ridge = 0 
        self.ridge_score = 0
        for i in range(0,len(alpha_values)):
            self.linridge= Ridge(alpha=alpha_values[i]).fit(self.X_train,self.y_train)
            print(f'For this value of alpha: {alpha_values[i]}, those are the scores:')
            print('R-squared score (training): {:.3f}'
            .format(self.linridge.score(self.X_train, self.y_train)))
            print('R-squared score (test): {:.3f} \n'
            .format(self.linridge.score(self.X_test, self.y_test)))
            if (self.linridge.score(self.X_test, self.y_test) > self.linridge.score(self.X_train, self.y_train)) and self.linridge.score(self.X_test, self.y_test) > self.ridge_score:
                self.best_alpha_ridge = alpha_values[i]
            self.linridge= Ridge(alpha=self.best_alpha_ridge).fit(self.X_train,self.y_train)

    def train_lasso(self) -> None:
        alpha_values=[1e-3,0.1,1,2,5,10,50,100]
        self.best_alpha_lasso = 0 
        self.lasso_score = 0
        for i in range(0,len(alpha_values)):
            self.linlasso= Lasso(alpha=alpha_values[i],max_iter=10000).fit(self.X_train,self.y_train)
            print(f'For this value of alpha: {alpha_values[i]}, those are the scores:')
            print('R-squared score (training): {:.3f}'
            .format(self.linlasso.score(self.X_train, self.y_train)))
            print('R-squared score (test): {:.3f} \n'
            .format(self.linlasso.score(self.X_test, self.y_test)))
            if (self.linlasso.score(self.X_test, self.y_test) > self.linlasso.score(self.X_train, self.y_train)) and self.linlasso.score(self.X_test, self.y_test) > self.lasso_score:
                self.best_alpha_lasso = alpha_values[i]
            self.linlasso= Lasso(alpha=self.best_alpha_lasso).fit(self.X_train,self.y_train)

    def poly_transform(self) -> None:
        """It's Possible to add some polynomial features to a linear regression. 
         Basically the number of variables or features will order the degree of polymonial expression.
         We can combine this features with normal regressions as Linear,Ridge and Lasso."""
        self.X_train_before = self.X_train
        self.X_test_before = self.X_test
        for i in range(1,8):
            self.poly = PolynomialFeatures(degree=i)
             #splitting the data with the addition of polynomial features
            self.X_train= self.poly.fit_transform(self.X_train)
            self.X_test = self.poly.fit_transform(self.X_test)
            print(f'For Polynomial features of degree {i} we have: \n')
            self.train_linreg()
            self.train_lasso()
            self.train_ridge()
    
    def normal_values(self) -> None:
        self.X_train = self.X_train_before
        self.X_test = self.X_test_before
    
    def train_mpl(self) -> None:
        self.mpl_score = 0
        self.best_alpha_mpl = 0
        for thisalpha in [100,110,125,135,150,200,500,1000]:
            for n,m in [(5,5),(5,10),(10,5),(25,25),(10,25),(25,50),(50,25),(50,50)]:
                self.mlpreg = MLPRegressor(hidden_layer_sizes = [n,m],
                                    activation = 'tanh',
                                    alpha = thisalpha,
                                    solver = 'lbfgs',max_iter=5000,random_state=0).fit(self.X_train, self.y_train)
                print(f'For the activation type "tanh" and setting the alpha regularization parameter as {thisalpha}, and layer sizes as [{n},{m}] the scores are:\
                \ntraining:{self.mlpreg.score(self.X_train,self.y_train)}\
                \t test:{self.mlpreg.score(self.X_test,self.y_test)} \n')
    

    



    

    

    

        
