import numpy as np
import matplotlib.pyplot as plt
import cPickle as p
def svd_plots():
    overall_dict = p.load(open('../stats/svd_results_rating.p','rb'))
    rmse_overall = overall_dict['RMSE']
    mae_overall = overall_dict['MAE']
    n_factors_overall = list(set(overall_dict['n_factors']))
    n_factors_overall.sort()
    n_epochs_overall = list(set(overall_dict['n_epochs']))
    n_epochs_overall.sort()
    print n_epochs_overall 
    rmse_10_factors = rmse_overall[0:10]
    rmse_50_factors = rmse_overall[10:20]
    rmse_100_factors = rmse_overall[20:30]
    
    print overall_dict
    plotLineChart("RMSE of SVD with overall ratings",n_epochs_overall,rmse_10_factors,rmse_50_factors,rmse_100_factors,"Number of epochs","RMSE","10 factors","50 factors","100 factors")
    
    
    
    combined_dict = p.load(open('../stats/svd_results_combined.p','rb'))
    rmse_combined = combined_dict['RMSE']
    mae_combined = combined_dict['MAE']
    n_factors_combined = list(set(combined_dict['n_factors']))
    n_epochs_combined = list(set(combined_dict['n_epochs']))
    n_factors_combined.sort()
    n_epochs_combined.sort()
    
    rmse_10_factors = rmse_combined[0:10]
    rmse_50_factors = rmse_combined[10:20]
    rmse_100_factors = rmse_combined[20:30]
    
    print overall_dict
    plotLineChart("RMSE of SVD with combined ratings",n_epochs_combined,rmse_10_factors,rmse_50_factors,rmse_100_factors,"Number of epochs","RMSE","10 factors","50 factors","100 factors")
    


    print combined_dict
def plotLineChart(title,indices,value1,value2,value3,xlabel,ylabel,label1,label2,label3):
    """
    Function to plot a line chart
    Args:
    title: The title of the figure
    indices: The indices on x-axis
    value1: the set of values 
    xlabel: the label of x axis
    ylabel: the label of y axis
    """
    plt.figure(figsize=(6,4))
    l1=plt.plot(indices,value1,'or-', linewidth=3,label=label1) #Plot the first series in red with circle marker2w
    l2=plt.plot(indices,value2,'xg-', linewidth=3,label=label2) #Plot the first series in red with circle marker2w
    l3=plt.plot(indices,value3,'+b-', linewidth=3,label=label3) #Plot the first series in red with circle marker2w
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("../Figures/\'"+title+"\'.png")
    plt.show()
    return;

svd_plots()
