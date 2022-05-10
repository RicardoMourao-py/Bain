from numpy import array
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
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score

__authors__ = ['Adney Costa','Matheus Oliveira','Nívea de Abreu','Ricardo Mourão']

class Pipeline:
    def __init__(self,df) -> None:
        print("[Start] Initializating the pipeline")
        self.df = df
        self.transformer=preprocessing.OrdinalEncoder()
    
    def split_data(self) -> None:
        print("[Loading] Spliting the data")
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
        self.X_train = self.df[(self.df['year']<2016) & (self.df['year']>1985)]
        self.X_test = self.df[(self.df['year']>2015) & (self.df['year']<2018)]
        self.y_train = self.X_train['destinated_area']
        self.y_test = self.X_test['destinated_area']
        self.X_train = self.X_train.drop('destinated_area',axis=1)
        self.X_test = self.X_test.drop('destinated_area',axis=1)
        # self.X_forecast = self.df[self.df['year']>2017]
        # self.y_forecast = self.X_forecast['destinated_area']
        # self.X_forecast = self.X_forecast.drop('destinated_area',axis=1)

    def encoder(self) -> None:
        print("[Loading] Enconding the features to use models")
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
        print(f"The best number of Neighbors is this shallow comparison is: {self.n_neighbors}")

        self.knnreg=KNeighborsRegressor(n_neighbors=self.n_neighbors).fit(self.X_train,self.y_train)

    def train_linreg(self) -> None:
        print('[Training] Training Linear Regression')
        self.linreg= LinearRegression().fit(self.X_train,self.y_train)
        print(f"[Linear Regression] The best score obtained was: {self.linreg.score(self.X_test, self.y_test)}")
        # print('R-squared score (training): {:.3f}'
        #     .format(self.linreg.score(self.X_train, self.y_train)))
        # print('R-squared score (test): {:.3f}'
        #     .format(self.linreg.score(self.X_test, self.y_test)))

    def train_ridge(self) -> None:
        print('[Training] Training Ridge Regression')
        alpha_values=[1e-3,0.1,1,2,5,10,50,100]
        self.best_alpha_ridge = 0 
        self.ridge_score = 0
        for i in range(0,len(alpha_values)):
            self.linridge= Ridge(alpha=alpha_values[i]).fit(self.X_train,self.y_train)
            # print(f'For this value of alpha: {alpha_values[i]}, those are the scores:')
            # print('R-squared score (training): {:.3f}'
            # .format(self.linridge.score(self.X_train, self.y_train)))
            # print('R-squared score (test): {:.3f} \n'
            # .format(self.linridge.score(self.X_test, self.y_test)))
            if self.linridge.score(self.X_test, self.y_test) > self.ridge_score:
                self.best_alpha_ridge = alpha_values[i]
                self.ridge_score = self.linridge.score(self.X_test, self.y_test)
        print(f"[Ridge] The best score obtained was: {self.ridge_score}")
        self.linridge= Ridge(alpha=self.best_alpha_ridge).fit(self.X_train,self.y_train)

    def train_lasso(self) -> None:
        print('[Training] Training Lasso Regression')
        alpha_values=[1e-3,0.1,1,2,5,10,50,100]
        self.best_alpha_lasso = 0 
        self.lasso_score = 0
        for i in range(0,len(alpha_values)):
            self.linlasso= Lasso(alpha=alpha_values[i],max_iter=10000).fit(self.X_train,self.y_train)
            # print(f'For this value of alpha: {alpha_values[i]}, those are the scores:')
            # print('R-squared score (training): {:.3f}'
            # .format(self.linlasso.score(self.X_train, self.y_train)))
            # print('R-squared score (test): {:.3f} \n'
            # .format(self.linlasso.score(self.X_test, self.y_test)))
            if self.linlasso.score(self.X_test, self.y_test) > self.lasso_score:
                self.lasso_score = self.linlasso.score(self.X_test, self.y_test)
                self.best_alpha_lasso = alpha_values[i]
        print(f"[Lasso] The best score obtained was: {self.lasso_score}")
        self.linlasso= Lasso(alpha=self.best_alpha_lasso).fit(self.X_train,self.y_train)

    def poly_transform(self) -> None:
        """It's Possible to add some polynomial features to a linear regression. 
         Basically the number of variables or features will order the degree of polymonial expression.
         We can combine this features with normal regressions as Linear,Ridge and Lasso."""
        self.X_train_before = self.X_train
        self.X_test_before = self.X_test
        for i in range(1,4):
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
        print('[Training] Training MPL Regression')
        self.mpl_score = 0
        self.best_alpha_mpl = 0
        self.n,self.m = 0,0
        for thisalpha in [135,200,500]:
            for n,m in [(5,5),(5,10),(10,5),(50,50)]:
                self.mlpreg = MLPRegressor(hidden_layer_sizes = [n,m],
                                    activation = 'tanh',
                                    alpha = thisalpha,
                                    solver = 'lbfgs',max_iter=50000,random_state=0).fit(self.X_train, self.y_train)
                if self.mlpreg.score(self.X_test,self.y_test) > self.mpl_score:
                    self.best_alpha_mpl = thisalpha
                    self.mpl_score = self.mlpreg.score(self.X_test,self.y_test)
                    self.n,self.m = n,m
                # print(f'For the activation type "tanh" and setting the alpha regularization parameter as {thisalpha}, and layer sizes as [{n},{m}] the scores are:\
                # \ntraining:{self.mlpreg.score(self.X_train,self.y_train)}\
                # \t test:{self.mlpreg.score(self.X_test,self.y_test)} \n')
        print(f"[MPL] The best score obtained was: {self.mpl_score}")
        self.mlpreg = MLPRegressor(hidden_layer_sizes = [self.n,self.m],
                                    activation = 'tanh',
                                    alpha = self.best_alpha_mpl,
                                    solver = 'lbfgs',max_iter=50000,random_state=0).fit(self.X_train, self.y_train)
    
    def train_models(self) -> None:
        print("[Loading] Training Models")
        self.train_lasso()
        self.train_linreg()
        self.train_ridge()
        self.train_mpl()
    
    def get_metrics(self) -> None:
        print('[Loading] Doing the preditions...')
        lasso_predict = self.linlasso.predict(self.X_test)
        ridge_predict = self.linridge.predict(self.X_test)
        linear_predict = self.linreg.predict(self.X_test)
        mpl_predict = self.mlpreg.predict(self.X_test)

        #perto do zero
        best_mse = 1
        mse_name = ''
        print('[mean_squared_error] For the models we get the scores (the best value possible is 0.0):')
        lassoMSE = mean_squared_error(self.y_test,lasso_predict)
        ridgeMSE = mean_squared_error(self.y_test,ridge_predict)
        linearMSE = mean_squared_error(self.y_test,linear_predict)
        mplMSE = mean_squared_error(self.y_test,mpl_predict)
        for model,mse in zip(['Lasso','Ridge','Linear','MPL'],[lassoMSE,ridgeMSE,linearMSE,mplMSE]):
            if mse<best_mse:
                mse_name = model
                best_mse = mse
            print(f'\t\t[Scores] {model} got {mse} as score')
        print(f'[Feedback]The best model in this metrics was: {mse_name}\n')

        #perto do zero
        best_mae = 1
        mae_name = ''
        print('[mean_absolute_error] For the models we get the scores (the best value possible is 0.0):')
        lassoMAE = mean_absolute_error(self.y_test,lasso_predict)
        ridgeMAE = mean_absolute_error(self.y_test,ridge_predict)
        linearMAE = mean_absolute_error(self.y_test,linear_predict)
        mplMAE = mean_absolute_error(self.y_test,mpl_predict)
        for model,mae in zip(['Lasso','Ridge','Linear','MPL'],[lassoMAE,ridgeMAE,linearMAE,mplMAE]):
            if mae<best_mae:
                mae_name = model
                best_mae = mae
            print(f'\t\t[Scores] {model} got {mae} as score')
        print(f'[Feedback]The best model in this metrics was: {mae_name}\n')

        #perto do 1
        best_evs = 0
        evs_name = ''
        print('[explained_variance_score] For the models we get the scores (the best value possible is 1.0):')
        lassoEVS = explained_variance_score(self.y_test,lasso_predict)
        ridgeEVS = explained_variance_score(self.y_test,ridge_predict)
        linearEVS = explained_variance_score(self.y_test,linear_predict)
        mplEVS = explained_variance_score(self.y_test,mpl_predict)
        for model,evs in zip(['Lasso','Ridge','Linear','MPL'],[lassoEVS,ridgeEVS,linearEVS,mplEVS]):
            if evs>best_evs:
                evs_name = model
                best_evs = evs
            print(f'\t\t[Scores] {model} got {evs} as score')
        print(f'[Feedback]The best model in this metrics was: {evs_name}\n')

        #perto do 1
        best_r2 = 0
        r2_name = ''
        print('[r2_score] For the models we get the scores (the best value possible is 1.0):')
        lassor2 = r2_score(self.y_test,lasso_predict)
        ridger2 = r2_score(self.y_test,ridge_predict)
        linearr2 = r2_score(self.y_test,linear_predict)
        mplr2 = r2_score(self.y_test,mpl_predict)
        for model,r2 in zip(['Lasso','Ridge','Linear','MPL'],[lassor2,ridger2,linearr2,mplr2]):
            if r2>best_r2:
                best_r2 = r2
                r2_name = model
            print(f'\t\t[Scores] {model} got {r2} as score')

        print(f'[Feedback]The best model in this metrics was: {r2_name}\n')

    def get_lin_prediction(self) -> array:
        return self.linreg.predict(self.X_test)

    def get_ridge_prediction(self) -> array:
        return self.linridge.predict(self.X_test)
    
    def get_lasso_prediction(self) -> array:
        return self.linlasso.predict(self.X_test)
    
    def get_mpl_prediction(self) -> array:
        return self.mlpreg.predict(self.X_test)
    

