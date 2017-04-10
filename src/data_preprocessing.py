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

def read_data_list(file_path,contains_header=True):
    '''
    Function to read data  in a csv fileand return as a list of lists

    ------
    Args:
    contains_header: Boolean variable indicating whether the file contains a header row (Optional, defaults to True).

    ------
    Returns:
    data: The data in the csv as a list of lists
    header: the header of the csv file. Returned only when header exists
    '''
    with open(DATA_DIR+file_path,'r') as f:
        reader=csv.reader(f)
        data=[row for row in reader]
        if contains_header:
            header=data.pop(0)
            return data,header
    return data

def create_train_test_split(data,train_ratio):
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
    np.random.shuffle(shuffle_indices)
    n_train_samples=np.floor(len(data)*train_ratio)
    train_indices=shuffle_indices[0:n_train_samples]
    test_indices=Shuffle_indices[n_train_samples:]
    train_data=data[train_indices]
    test_data=data[test_indices]
    return train_data,test_data
    



def main():
    if len(sys.argv)>1:
        df=read_raw_data(sys.argv[1],use_dump=True)
    else:
        df=read_raw_data()

    # print df
    print "==================================="
    print "Read DF"
    print "==================================="
    df2=compute_sentiment_scores(df,use_dump=True)
    print("Scores calculated")
    feature_list=['reviewerID','asin','overall','unixReviewTime']
    df3=extract_features(df2,feature_list,'data_collab.csv')
    print df3
    return

if __name__=='__main__':
    main()



# dataFrame = read_data()
# newDataFrame = computeSentimentScores(dataFrame)
