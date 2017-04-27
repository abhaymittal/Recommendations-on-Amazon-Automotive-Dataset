from __future__ import print_function
from surprise import BaselineOnly
from surprise.dataset import DatasetAutoFolds
from surprise.dataset import Dataset
from surprise import evaluate
from surprise import Reader
from surprise import SVD
from surprise import NMF
from surprise import KNNWithMeans
from surprise import GridSearch
from surprise import accuracy
import numpy as np
import data_preprocessing as dp
import pprint
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
            grid_search.evaluate(validation_set)
            best_model_RMSE = grid_search.best_params['RMSE']
            validation_rmse = grid_search.best_score['RMSE']
            best_model_mae = grid_search.best_params['MAE']
            validation_mae = grid_search.best_score['MAE']
            #print(validation_rmse)
            #print(validation_mae)
            print(type(grid_search.cv_results))
            print(grid_search.cv_results)
            
            #Test based on best training RMSE
            n_epochs = best_model_RMSE['n_epochs']
            n_factors = best_model_RMSE['n_factors']
            lr_all = best_model_RMSE['lr_all']
            reg_all = best_model_RMSE['reg_all']
            algo = SVD(n_epochs = n_epochs, n_factors = n_factors,lr_all = lr_all, reg_all = reg_all)
            algo.train(train_set)
            predictions = algo.test(test_set)
            test_rmse = accuracy.rmse(predictions,verbose = True)
            test_mae = accuracy.mae(predictions,verbose = True)
            print("RMSE of predictions",test_rmse)
            print("MAE of predictions",test_mae)
        
        if algorithm == 'NMF':
            
            param_grid = {'n_epochs':np.arange(1,2).tolist(),'n_factors':np.arange(1,2).tolist()}
            grid_search = GridSearch(NMF, param_grid, measures=['RMSE', 'MAE'])
            grid_search.evaluate(validation_set)
            best_model_RMSE = grid_search.best_params['RMSE']
            validation_rmse = grid_search.best_score['RMSE']
            best_model_mae = grid_search.best_params['MAE']
            validation_mae = grid_search.best_score['MAE']
            print(validation_rmse)
            print(validation_mae)
            
            #Test based on best training RMSE
            n_epochs = best_model_RMSE['n_epochs']
            n_factors = best_model_RMSE['n_factors']
            algo = NMF(n_epochs = n_epochs, n_factors = n_factors)
            algo.train(train_set)
            predictions = algo.test(test_set)
            test_rmse = accuracy.rmse(predictions,verbose = True)
            test_mae = accuracy.mae(predictions,verbose = True)
            print("RMSE of predictions",test_rmse)
            print("MAE of predictions",test_mae)
        
        if algorithm == 'KNNWithMeans':
            param_grid = {'k':np.arange(1,20).tolist(),'sim_options':[{'name':'cosine','user_based':True},
                {'name':'msd','user_based':True},{'name':'pearson','user_based':True}]}
            grid_search = GridSearch(KNNWithMeans,param_grid,measures=['RMSE','MAE'])
            grid_search.evaluate(validation_set)

            best_model_RMSE = grid_search.best_params['RMSE']
            validation_rmse = grid_search.best_score['RMSE']
            best_model_mae = grid_search.best_score['MAE']
            validation_mae = grid_search.best_score['MAE']

            #Test based on best training RMSE
            k = best_model_RMSE['k']
            sim_options = best_model_RMSE['sim_options']
            algo = KNNWithMeans(k = k, sim_options = sim_options)
            algo.train(train_set)
            predictions = algo.test(test_set)
            test_rmse = accuracy.rmse(predictions,verbose = True)
            test_mae = accuracy.mae(predictions,verbose = True)
            print("RMSE of predictions",test_rmse)
            print("MAE of predictions",test_mae)

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
    data_sentiment = dp.read_data_list('data_collab_sentiment.csv',contains_header = False)
    data_combined = dp.read_data_list('data_collab_combined.csv',contains_header = False)
    
    for d in data:
        d[2] = float(d[2])
    
    for d in data_sentiment:
        d[2] = float(d[2])
    
    for d in data_combined:
        d[2] = float(d[2])


    print("Size of data = ",len(data))
    reader = Reader(line_format='user item rating timestamp', sep=',')
    sp=Surprise_recommender(reader)
    
    #Create splits
    train,test=dp.create_train_test_split(data,0.8)
    train_sentiment,test_sentiment=dp.create_train_test_split(data_sentiment,0.8)
    train_combined,test_combined = dp.create_train_test_split(data_combined,0.8)
    
    #Create validation sets
    validation = DatasetForCV(train,None,reader)
    validation_sentiment  = DatasetForCV(train_sentiment,None, reader)
    validation_combined = DatasetForCV(train_combined,None,reader)
    
    #Create train and test sets
    train=sp.create_train_set(train)
    test=sp.create_test_set(test)
    train_sentiment = sp.create_train_set(train_sentiment)
    test_sentiment = sp.create_test_set(test_sentiment)
    train_combined = sp.create_train_set(train_combined)
    test_combined = sp.create_test_set(test_combined)
    
    #sp.train_test_model(validation,train,test,'NMF')
    sp.train_test_model(validation_combined,train_combined,test_combined,'SVD')
    #sp.train_test_model(validation,train,test,'KNNWithMeans')
    #sp.train_test_model(validation_sentiment,train_sentiment,test_sentiment,'SVD')
    # print(train)
    # for t in test:
    #     print(t)
    # return 


if __name__=='__main__':
    main()
