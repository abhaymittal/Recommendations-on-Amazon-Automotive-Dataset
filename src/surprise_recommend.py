from __future__ import print_function
from surprise import BaselineOnly
from surprise import Dataset
from surprise import evaluate
from surprise import Reader

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



def main():
    data=dp.read_data_list('data_collab.csv',contains_header=False)
    print("Size of data = ",len(data))
    reader = Reader(line_format='user item rating timestamp', sep=',')
    sp=Surprise_recommender(reader)
    train,test=dp.create_train_test_split(data,0.8)
    train=sp.create_train_set(train)
    test=sp.create_test_set(test)
    # print(train)
    # for t in test:
    #     print(t)
    # return 


if __name__=='__main__':
    main()
