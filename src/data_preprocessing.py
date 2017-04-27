from __future__ import print_function
import pandas as pd
try:
    import cPickle as p
except:
    import Pickle as p
import os
from textblob import TextBlob
import numpy as np
import sys
import csv


DATA_DIR='../data/'

def read_raw_data(file_name,use_dump=True):
    '''
    Parse the raw data file and convert it into a Pandas DataFrame.
    ------
    Args:
    file_name: The name of the file. 
    use_dump: A boolean variable. A value of true means the function will try to load a pre-saved pickle dump and return it
    ------
    Returns:
    dataFrame: A pandas data frame for the data
    '''
    #If pickle dump doesn't already exist, parse the file and create one
    if not (use_dump and 'mainDataFrame.p' in os.listdir(DATA_DIR)):
        dataFrame=pd.read_json(path_or_buf =DATA_DIR+file_name,lines=True)
        p.dump(dataFrame,open(DATA_DIR+'mainDataFrame.p','wb'))
    #Else return the dataframe from the existing dump
    else:
        dataFrame = p.load(open(DATA_DIR+'mainDataFrame.p','rb'))
    return dataFrame


def compute_sentiment_scores(data_frame,use_dump=True,target_file=None):
    '''
    Function to compute sentiment polarity scores using TextBlob(Default stuff for now)
    ------
    Args:
    data_frame: The dataframe containing the reviews
    use_dump: A boolean variable. True means the function will try to load a pre-saved pickle dump and return it
    target_file: Optional. File to save the data frame containing calculated reviews.
    ------
    Returns: 
    data_frame: DataFrame with sentiment scores appended
    '''
    #If pickle dump doesn't already exist, compute scores and append to existing dataframe
    if not(use_dump and 'mainDataFrameWithSentimentScores.p' in os.listdir('../data')):
        listOfScores = np.zeros(len(data_frame['reviewText']),dtype='float32')
        i=0
        for review in data_frame['reviewText']:
            reviewBlob = TextBlob(review)
            #print review,reviewBlob.sentiment
            listOfScores[i]=reviewBlob.sentiment.polarity
            i=i+1
            
        # seriesOfScores = pd.Series(listOfScores)
        # newDataFrame = data_frame.append(seriesOfScores,ignore_index = True)
        data_frame['scores']=pd.Series(listOfScores)
        p.dump(data_frame,open('../data/mainDataFrameWithSentimentScores.p','wb'))
        if target_file is not None:
            data_frame.to_csv(DATA_DIR+target_file)
    #If the pickle dump already exists, return the dataframe with the computed sentiment scores
    else:
        data_frame = p.load(open('../data/mainDataFrameWithSentimentScores.p','rb'))

    return data_frame


def extract_features(data_frame,feature_list,target_file=None,header=False,index=False):
    '''
    Function to extract some columns from a data frame and return them
    ------
    Args:
    data_frame: The input data frame
    feature_list: A list containing the column/feature names to extract
    target_file: File to save the new data frame to. Optional
    header: Boolean variable indicating whether to save the header in the csv file (Optional, defautls to False)
    index: Boolean variable indicating whether to save the index of the data frames  (Optional, defautls to False)

    ------
    Returns:
    df: data frame consisting of only the subset of the features
    '''
    # Extract the columns into a new data frame
    df=data_frame[feature_list]

    # Save the data frame if required
    if target_file is not None:
        df.to_csv(DATA_DIR+target_file,index=index,header=header)
    return df

def read_data_list(file_name,contains_header=True):
    '''
    Function to read data  in a csv fileand return as a list of lists

    ------
    Args:
    file_name: the name of the file containing data
    contains_header: Boolean variable indicating whether the file contains a header row (Optional, defaults to True).

    ------
    Returns:
    data: The data in the csv as a list of lists
    header: the header of the csv file. Returned only when header exists
    '''
    with open(DATA_DIR+file_name,'r') as f:
        reader=csv.reader(f)
        data=[row for row in reader]
        if contains_header:
            header=data.pop(0)
            return data,header
    return data

def create_train_test_split(data,train_ratio,train_indices=None,test_indices=None):
    '''
    Function to create training and test splits from the data
    
    ------
    Args:
    data: The input dataset as list of lists
    train_ratio: The ratio of training data 
    
    ------
    Returns:
    train_data: list of lists containing training samples
    test_data: list of lists containing testing samples
    '''


    shuffle_indices=np.arange(len(data))
    flag=False
    if train_indices is None:
        flag=True
        np.random.shuffle(shuffle_indices)
        n_train_samples=int(np.floor(len(data)*train_ratio))
        train_indices=shuffle_indices[0:n_train_samples]
        test_indices=shuffle_indices[n_train_samples:]
    train_data=[data[i] for i in train_indices]
    test_data=[data[i] for i in test_indices]
    if flag:
        return train_data,test_data,train_indices,test_indices
    else:
        return train_data,test_data
    
def compute_combined_score(data_frame):
    '''
    Function to compute combined score from sentiment score and overall rating

    ------
    Args:
    data_frame: Pandas dataframe containing sentiment scores and reviews

    ------
    Returns:
    data_frame: Pandas dataframe containing combined scores
    '''
    ratings = np.array(data_frame['overall'])
    sentiment_scores = np.array(data_frame['scores'])
    #print(type(ratings))
    #print(type(sentiment_scores))
    ratings_sd = np.std(ratings)
    sentiment_scores_sd = np.std(sentiment_scores)
    mean_ratings = np.mean(ratings)
    mean_sentiment_scores = np.mean(sentiment_scores)
    
    combined_ratings = []
    
    for rating,sentiment_score in zip(ratings,sentiment_scores):
        scaled_rating = (float(rating) - mean_ratings)/ratings_sd
        scaled_sentiment_score = (float(sentiment_score) - mean_sentiment_scores)/sentiment_scores_sd
        combined_ratings.append(scaled_rating+scaled_sentiment_score)
    print(np.mean(np.array(combined_ratings)),np.std(np.array(combined_ratings)))
    data_frame['combinedScore'] = pd.Series(combined_ratings)
    
    return data_frame



def main():
    if len(sys.argv)>1:
        df=read_raw_data(sys.argv[1],use_dump=True)
    else:
        df=read_raw_data()

    print("===================================")
    print("Read DF")
    print("===================================")
    df2=compute_sentiment_scores(df,use_dump=True)
    #print(df2.dtypes)
    print("Scores calculated")
    df3 = compute_combined_score(df2)
    print(df3.dtypes)
    feature_list=['reviewerID','asin','overall','unixReviewTime']
    feature_list_sentiment_scores = ['reviewerID','asin','scores','unixReviewTime']
    feature_list_combined = ['reviewerID','asin','combinedScore','unixReviewTime']
    df4 = extract_features(df2,feature_list_sentiment_scores,'data_collab_sentiment.csv')
    df5=extract_features(df2,feature_list,'data_collab.csv')
    df6 = extract_features(df3,feature_list_combined,'data_collab_combined.csv')
    return

if __name__=='__main__':
    main()



# dataFrame = read_data()
# newDataFrame = computeSentimentScores(dataFrame)
