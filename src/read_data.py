import pandas as pd
try:
    import cPickle as p
except:
    import Pickle as p
import os
from textblob import TextBlob

'''
Parse the data file and convert it into a Pandas DataFrame.
Return value: DataFrame 
'''
def read_data():

    #If pickle dump doesn't already exist, parse the file and create one
    if not 'mainDataFrame.p' in os.listdir('../data'):
        dataFrame=pd.read_json(path_or_buf ='../data/dataFile.json',lines=True)
        p.dump(dataFrame,open('../data/mainDataFrame.p','wb'))
    #Else return the dataframe from the existing dump
    else:
        dataFrame = p.load(open('../data/mainDataFrame.p','rb'))
    return dataFrame

'''
Compute sentiment polarity scores using TextBlob(Default stuff for now)
Return value: DataFrame with sentiment scores appended
'''
def computeSentimentScores(dataFrame):
    #If pickle dump doesn't already exist, compute scores and append to existing dataframe
    if not 'mainDataFrameWithSentimentScores.p' in os.listdir('../data'):
        listOfScores = []
        for review in dataFrame['reviewText']:
            reviewBlob = TextBlob(review)
            #print review,reviewBlob.sentiment
            listOfScores.append(reviewBlob.sentiment)
        seriesOfScores = pd.Series(listOfScores)
        newDataFrame = dataFrame.append(seriesOfScores,ignore_index = True)
        p.dump(newDataFrame,open('../data/mainDataFrameWithSentimentScores.p','wb'))
    #If the pickle dump already exists, return the dataframe with the computed sentiment scores
    else:
        newDataFrame = p.load(open('../data/mainDataFrameWithSentimentScores.p','rb'))

    return newDataFrame



dataFrame = read_data()
newDataFrame = computeSentimentScores(dataFrame)
