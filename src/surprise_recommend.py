from __future__ import print_function
from surprise import BaselineOnly
from surprise.dataset import DatasetAutoFolds
from surprise.dataset import Dataset
from surprise import evaluate
from surprise import Reader
from surprise import SVD
from surprise import GridSearch
from surprise import accuracy
import numpy as np
import data_preprocessing as dp

# reader = Reader(line_format='user item rating timestamp', sep=',')

# data = Dataset.load_from_file('../data_collab.csv', reader=reader)

class Surprise_recommender:
    def __init__(self,reader):
        '''
        Constructor

        ------
        Args:
        reader: A reader object for the dataset object in surprise
        '''
        self.reader=reader
        return 

    def create_test_set(self,test_data):
        '''
        Function to create test_set
        This function drops timestamp from the data

        ------
        Args:
        test_data: input test data
        
        ------
        Returns:
        ts: test data after removing time stamp feature
        '''
        ts=[[td[0],td[1],td[2]] for td in test_data]
        return ts


    def create_train_set(self,train_data):
        '''
        Function to create training set
        
        ------
        Args:
        train_data: Training set in the form of list
        
        ------
        Returns:
        Trainset object from surprise
        '''
        ds=Dataset(self.reader)
        return ds.construct_trainset(train_data)
    
    def train_test_model(self,validation_set,train_set,test_set,algorithm):
        '''
        Function to train models using different algorithms

        ------
        Args:
        train_set: The training data formatted according to the needs of surprise
        algorithm: The algorithm for training the model

        ------
        Returns:Model that can be evaluated on the test set
        '''
        if algorithm == 'SVD':
            
            param_grid = {'n_epochs':np.arange(1,2).tolist(),'n_factors':np.arange(1,2).tolist(),'lr_all':[0.001,0.002],'reg_all':[0.1,0.2]}
            grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'MAE'])
            grid_search.evaluate(data_set)
            best_model_RMSE = grid_search.best_params['RMSE']
            training_rmse = grid_search.best_score['RMSE']
            best_model_mae = grid_search.best_params['MAE']
            training_mae = grid_search.best_score['MAE']
            print(training_RMSE)
            print(training_mae)
            
            '''
            #Test based on best training RMSE
            n_epochs = best_model_RMSE['n_epochs']
            n_factors = best_model_RMSE['n_factors']
            lr_all = best_model_RMSE['lr_all']
            reg_all = best_model_RMSE['reg_all']
            algo = SVD(n_epochs = n_epochs, n_factors = n_factors,lr_all = lr_all, reg_all = reg_all)
            predictions = algo.test(test_set)
            '''

class DatasetForCV(DatasetAutoFolds):
    
    def __init__(self,raw_ratings,ratings_file = None,reader = None):
        #self g.ratings_file = ratings_file
        self.reader = reader
        self.n_folds = 5
        self.shuffle = False
        self.raw_ratings = raw_ratings
        self.ratings_file=ratings_file

def main():
    data=dp.read_data_list('data_collab.csv',contains_header=False)
    for d in data:
        d[2] = float(d[2])
    print("Size of data = ",len(data))
    reader = Reader(line_format='user item rating timestamp', sep=',')
    sp=Surprise_recommender(reader)
    train,test=dp.create_train_test_split(data,0.8)
    validation = DatasetForCV(train,None,reader)
    train=sp.create_train_set(train)
    test=sp.create_test_set(test)
    sp.train_test_model(validation,train,test,'SVD')
    # print(train)
    # for t in test:
    #     print(t)
    # return 


if __name__=='__main__':
    main()
