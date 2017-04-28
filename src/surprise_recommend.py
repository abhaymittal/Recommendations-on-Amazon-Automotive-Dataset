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
from operator import itemgetter
import numpy as np
import data_preprocessing as dp
import pprint
import datetime as dt
import os
try:
    import cPickle as p
except:
    import Pickle as p
# reader = Reader(line_format='user item rating timestamp', sep=',')

# data = Dataset.load_from_file('../data_collab.csv', reader=reader)
DATA_DIR='../data/'

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
        Basically a list with the following format: user, item, rating
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
        Basically a list with the following format: user, item, rating, timestamp
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
            
            param_grid = {'n_epochs':np.arange(0,100,90).tolist(),'n_factors':[10]}
            grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'MAE'])
            
            start = dt.datetime.now()
            grid_search.evaluate(validation_set)
            end = dt.datetime.now()

            time_taken = ((end-start).microseconds)/(1000000*60)
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
            
            param_grid = {'n_epochs':np.arange(0,100,10).tolist(),'n_factors':[10,100]}
            grid_search = GridSearch(NMF, param_grid, measures=['RMSE', 'MAE'])
            start = dt.datetime.now() 
            grid_search.evaluate(validation_set)
            end = dt.datetime.now()
            time_taken = ((end-start).microseconds)/(1000000*60)
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
            start = dt.datetime().now
            grid_search.evaluate(validation_set)
            end = dt.datetime.now()
            time_taken = ((end-start).microseconds)/(1000000*60)
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
            return time_taken

    def generate_top_n_recommendation(self,test_set,train_set):
        '''
        Function to generate top N recommendations
        
        ----
        Args:
        user_id: The id of the user
        test_set: The testing set as a list
        train_set: The training set as a list
        '''
        user_list=set([x[0] for x in train_set])
        print ("Number of users = ",len(user_list))
        
        precision_list=[]
        recall_list=[]
        f_score_list=[]
        j=0
        for user in user_list:
            # print("===============================================================")
            # print("=====================+++++++++++++++++++++++===================")
            # print("===============================================================")
            j+=1
            if j%1000==0:
                print("Touchdown, j = ",j)
            item_train=set([x[1] for x in train_set if x[0]==user])
            item_test=set([x[1] for x in test_set if x[0]==user])
            item_train_all=set([x[1] for x in train_set])
            item_test_all=set([x[1] for x in test_set])
            item_all=item_train_all.union(item_test_all)
            # print("User = ",user)
            # print("===============================================================")
            # print("TRain items = ",item_train)
            # print("===============================================================")
            # print("Test items = ",item_test)
            # print("ITem all = ",item_all)
            # print("Number of  test items= ",len(item_test))
            negative_items=[x for x in item_all if x not in item_train and x not in item_test]
            # print("Number of negative items = ",len(negative_items))

            # Get 1000 random negative items
            negative_indices=np.random.randint(0,len(negative_items),size=1000)
            negative_subset=[negative_items[x] for x in negative_indices]
            # Get 5 positive items from testing set:
            positive_subset=list(item_test)
            np.random.shuffle(positive_subset)
            # print("Positive subset items = ",positive_subset)
            # print(negative_subset)
            subset=positive_subset+negative_subset
            pred_list=[]
            for item in subset:
                pred=self.algo.predict(user,item,r_ui=1,verbose=False)
                pred_list.append(pred)
            predictions = sorted(pred_list, key=lambda x: x.est,reverse=True)
            # print(" =============================================================")
            precision=self.calculate_precision(predictions,positive_subset,10)
            # print("Precision = ",precision)
            recall=self.calculate_recall(predictions,positive_subset,10)
            # print("Recall = ",recall)
            # f_score=self.calculate_f_measure(precision,recall)
            # print("F score = ",f_score)
            precision_list.append(precision)
            recall_list.append(recall)
            # f_score_list.append(f_score)

        precision=np.mean(precision_list)
        recall=np.mean(recall_list)
        print ("Mean precision = ",precision)
        print("Mean recall = ",recall)
        print("fscore=",self.calculate_f_measure(precision,recall))
        return

    def calculate_precision(self,predictions,positive_items,N):
        '''
        Function to calculate precision
        '''
        count=0
        for i in np.arange(N):
            p=predictions[i]
            if p.iid in positive_items:
                count+=1
        precision=float(count)/N
        return precision


    def calculate_recall(self,predictions,positive_items,N):
        '''
        Function to calculate recall
        '''
        count=0
        pred=predictions[:N] #Get TOP N Predictions
        for p in positive_items:
            for i in pred:
                if i.iid==p:
                    count+=1
                    break

        recall=float(count)/len(positive_items)
        return recall

    def  calculate_f_measure(self,precision,recall):
        '''
        Function to calculate recall
        '''
        try:
            f=2.0*precision*recall/(precision+recall)
        except:
            f=0
        return f

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

    #a = dict()
    print("Size of data = ",len(data))
    reader = Reader(line_format='user item rating timestamp', sep=',')
    sp=Surprise_recommender(reader)


    #Create splits
    if not('train_test_splits.p' in os.listdir(DATA_DIR)):
        train,test,train_indices,test_indices=dp.create_train_test_split(data,0.8)
        train_sentiment,test_sentiment=dp.create_train_test_split(data_sentiment,0.8,train_indices,test_indices)
        train_combined,test_combined= dp.create_train_test_split(data_combined,0.8,train_indices,test_indices)
        print("Creating pickle dump")
        with open(DATA_DIR+'train_test_splits.p','wb') as f:
            p.dump([train_indices, test_indices, train,test,train_sentiment,test_sentiment,train_combined,test_combined],f)
    else:
        print("Load pickle dump")
        with open(DATA_DIR+'train_test_splits.p','rb') as f:
            train_indices, test_indices, train,test,train_sentiment,test_sentiment,train_combined,test_combined=p.load(f)
    #Create validation sets
    validation = DatasetForCV(train,None,reader)
    validation_sentiment  = DatasetForCV(train_sentiment,None, reader)
    validation_combined = DatasetForCV(train_combined,None,reader)


    # #Create train and test sets
    train_list=train
    train=sp.create_train_set(train)
    test_list=test
    test=sp.create_test_set(test)
    # print(train)
    # print(test)
    train_sent_list=train_sentiment
    train_sentiment = sp.create_train_set(train_sentiment)
    test_sent_list=test_sentiment
    test_sentiment = sp.create_test_set(test_sentiment)
    train_combined = sp.create_train_set(train_combined)
    test_combined = sp.create_test_set(test_combined)


    #Testing and trainig models based on RMSE
    
    #Run and measure RMSE, MAE for different algorithms
    print('--------------Normal Ratings---------------------')
    time_normal_svd = sp.train_test_model(validation,train,test,'SVD','rating')
    sp.generate_top_n_recommendation(test_list,train_list)
    '''
    print('--------------Combined Scores---------------------')
    time_combined_svd = sp.train_test_model(validation_combined,train_combined,test_combined,'SVD','combined')
    print('--------------Sentiment Scores---------------------')
    time_sentiment_svd = sp.train_test_model(validation_sentiment,train_sentiment,test_sentiment,'SVD','sentiment')
    ''' 
    #  #Run and measure RMSE, MAE for different algorithms
    
    # print('--------------Normal Ratings---------------------')
    # time_normal_nmf = sp.train_test_model(validation,train,test,'NMF','rating')
    # print('--------------Combined Scores---------------------')
    # time_combined_nmf = sp.train_test_model(validation_combined,train_combined,test_combined,'NMF','combined')
    # print('--------------Sentiment Scores---------------------')
    # time_sentiment_nmf = sp.train_test_model(validation_sentiment,train_sentiment,test_sentiment,'NMF','sentiment')
    # '''
    # print('--------------Normal Ratings---------------------')
    # time_normal_knn = sp.train_test_model(validation,train,test,'KNNWithMeans','rating')
    # print('--------------Combined Scores---------------------')
    # time_combined_knn = sp.train_test_model(validation_combined,train_combined,test_combined,'KNNWithMeans','combined')
    # print('--------------Sentiment Scores---------------------')
    # time_sentiment_knn = sp.train_test_model(validation_sentiment,train_sentiment,test_sentiment,'KNNWithMeans','sentiment')
    # '''
    # '''
    # sp.train_test_model(validation,train,test,'NMF','rating')
    # sp.train_test_model(validation_combined,train_combined,test_combined,'NMF','combined')
    # sp.train_test_model(validation_sentiment,train_sentiment,test_sentiment,'NMF','sentiment')
    
    # sp.train_test_model(validation,train,test,'KNNWithMeans','rating')
    # sp.train_test_model(validation_combined,train_combined,test_combined,'KNNWithMeans','combined')
    # sp.train_test_model(validation_sentiment,train_sentiment,test_sentiment,'KNNWithMeans','sentiment')
    # '''
    # print(train)
    # for t in test:
    #     print(t)
    # return 


if __name__=='__main__':
    main()
