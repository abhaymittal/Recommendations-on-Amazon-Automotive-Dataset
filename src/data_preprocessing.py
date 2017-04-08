import pandas as pd
try:
    import cPickle as p
except:
    import Pickle as p
import os
from textblob import TextBlob
import numpy as np
import sys

DATA_DIR='../data/'
'''
Parse the data file and convert it into a Pandas DataFrame.
Return value: DataFrame 
'''
def read_data(file_name='',use_dump=True):

    #If pickle dump doesn't already exist, parse the file and create one
    if not (use_dump and 'mainDataFrame.p' in os.listdir(DATA_DIR)):
        dataFrame=pd.read_json(path_or_buf =DATA_DIR+file_name,lines=True)
        p.dump(dataFrame,open(DATA_DIR+'mainDataFrame.p','wb'))
    #Else return the dataframe from the existing dump
    else:
        dataFrame = p.load(open(DATA_DIR+'mainDataFrame.p','rb'))
    return dataFrame

'''
Compute sentiment polarity scores using TextBlob(Default stuff for now)
Return value: DataFrame with sentiment scores appended
'''
def compute_sentiment_scores(data_frame,use_dump=True):
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
        data_frame.to_csv('abc.csv')
    #If the pickle dump already exists, return the dataframe with the computed sentiment scores
    else:
        data_frame = p.load(open('../data/mainDataFrameWithSentimentScores.p','rb'))

    return data_frame


def main():
    if len(sys.argv)>0:
        df=read_data(sys.argv[1],use_dump=False)
    else:
        df=read_data()

    print df
    print "==================================="
    print "Read DF"
    print "==================================="
    df2=compute_sentiment_scores(df,use_dump=False)
    print("Scores calculated")
    # print df2['scores']
    # print df2
    return

if __name__=='__main__':
    main()



# dataFrame = read_data()
# newDataFrame = computeSentimentScores(dataFrame)
