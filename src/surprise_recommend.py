from __future__ import print_function
from surprise import BaselineOnly
import cPickle as p
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
    
    def train_test_model(self,validation_set,train_set,test_set,algorithm,task):
        '''
        Function to train models using different algorithms

        ------
        Args:
        train_set: The training data formatted according to the needs of surprise
        algorithm: The algorithm for training the model
        test_set: Testing data to check RMSE and MAE after GridSearch
        validation_set: Dataset for hyperparameter optimization
        task: Make predictions for rating, sentiment scores or for combined rating

        ------
        Returns:Model that can be evaluated on the test set
        '''
        
        if algorithm == 'SVD':
            
            param_grid = {'n_epochs':np.arange(0,100,50).tolist(),'n_factors':[10,100]}
            grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'MAE'])
            grid_search.evaluate(validation_set)
            p.dump(grid_search,open('../stats/svd_results_'+task+'.p','wb'))
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
            self.algo = SVD(n_epochs = n_epochs, n_factors = n_factors)
            self.algo.train(train_set)
            predictions = self.algo.test(test_set)
            test_rmse = accuracy.rmse(predictions,verbose = True)
            test_mae = accuracy.mae(predictions,verbose = True)
            print("RMSE of predictions",test_rmse)
            print("MAE of predictions",test_mae)
        
        if algorithm == 'NMF':
            
            param_grid = {'n_epochs':np.arange(1,100,10).tolist(),'n_factors':[10,100]}
            grid_search = GridSearch(NMF, param_grid, measures=['RMSE', 'MAE'])
            grid_search.evaluate(validation_set)
            p.dump(grid_search,open('../stats/nmf_results_'+task+'.p','wb'))
            best_model_RMSE = grid_search.best_params['RMSE']
            validation_rmse = grid_search.best_score['RMSE']
            best_model_mae = grid_search.best_params['MAE']
            validation_mae = grid_search.best_score['MAE']
            print(validation_rmse)
            print(validation_mae)
            
            #Test based on best training RMSE
            n_epochs = best_model_RMSE['n_epochs']
            n_factors = best_model_RMSE['n_factors']
            self.algo = NMF(n_epochs = n_epochs, n_factors = n_factors)
            self.algo.train(train_set)
            predictions = self.algo.test(test_set)
            test_rmse = accuracy.rmse(predictions,verbose = True)
            test_mae = accuracy.mae(predictions,verbose = True)
            print("RMSE of predictions",test_rmse)
            print("MAE of predictions",test_mae)
        
        if algorithm == 'KNNWithMeans':
            param_grid = {'k':np.arange(1,20).tolist(),'sim_options':[{'name':'cosine','user_based':True},
                {'name':'msd','user_based':True},{'name':'pearson','user_based':True}]}
            grid_search = GridSearch(KNNWithMeans,param_grid,measures=['RMSE','MAE'])
            grid_search.evaluate(validation_set)
            p.dump(grid_search,open('../stats/knn_means_results'+task+'.p','wb'))
    
            best_model_RMSE = grid_search.best_params['RMSE']
            validation_rmse = grid_search.best_score['RMSE']
            best_model_mae = grid_search.best_score['MAE']
            validation_mae = grid_search.best_score['MAE']

            #Test based on best training RMSE
            k = best_model_RMSE['k']
            sim_options = best_model_RMSE['sim_options']
            self.algo = KNNWithMeans(k = k, sim_options = sim_options)
            self.algo.train(train_set)
            predictions = self.algo.test(test_set)
            test_rmse = accuracy.rmse(predictions,verbose = True)
            test_mae = accuracy.mae(predictions,verbose = True)
            print("RMSE of predictions",test_rmse)
            print("MAE of predictions",test_mae)
            return

        def generate_top_n_recommendation(self,user_id,item_set,test_item_set,train_item_set):
            '''
            Function to generate top N recommendations

            ----
            Args:
            user_id: The id of the user
            item_set: The set of all the items available
            test_item_set: The set of items in the testing set
            train_item_set: The set of items in the training set
            '''
            return

class DatasetForCV(DatasetAutoFolds):
    
    def __init__(self,raw_ratings,ratings_file = None,reader = None):
        #self g.ratings_file = ratings_file
        self.reader = reader
        self.n_folds = 5
        self.shuffle = False
        self.raw_ratings = raw_ratings
        self.ratings_file=ratings_file

def main():
    
    #Load the data
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
    train,test,train_indices,test_indices=dp.create_train_test_split(data,0.8)
    train_sentiment,test_sentiment=dp.create_train_test_split(data_sentiment,0.8,train_indices,test_indices)
    train_combined,test_combined= dp.create_train_test_split(data_combined,0.8,train_indices,test_indices)
    
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
    
    #Run and measure RMSE, MAE for different algorithms
    print('--------------Normal Ratings---------------------')
    sp.train_test_model(validation,train,test,'SVD','rating')
    # print('--------------Combined Scores---------------------')
    # sp.train_test_model(validation_combined,train_combined,test_combined,'SVD','combined')
    # print('--------------Sentiment Scores---------------------')
    # sp.train_test_model(validation_sentiment,train_sentiment,test_sentiment,'SVD','sentiment')
    
    '''
    sp.train_test_model(validation,train,test,'NMF','rating')
    sp.train_test_model(validation_combined,train_combined,test_combined,'NMF','combined')
    sp.train_test_model(validation_sentiment,train_sentiment,test_sentiment,'NMF','sentiment')
    
    sp.train_test_model(validation,train,test,'KNNWithMeans','rating')
    sp.train_test_model(validation_combined,train_combined,test_combined,'KNNWithMeans','combined')
    sp.train_test_model(validation_sentiment,train_sentiment,test_sentiment,'KNNWithMeans','sentiment')
    '''
    # print(train)
    # for t in test:
    #     print(t)
    # return 


if __name__=='__main__':
    main()
