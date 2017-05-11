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
    return

def knn_plots():
    overall_dict=p.load(open('../stats/knn_means_resultsrating.p'))
    overall_dict=overall_dict.cv_results
    rmse_overall=overall_dict['RMSE']
    mae_overall=overall_dict['MAE']
    params_overall=overall_dict['params']
    k_overall=list(set(overall_dict['k']))
    k_overall.sort()
    n_k=len(k_overall)
    rmse_cosine=rmse_overall[0:n_k]
    rmse_msd=rmse_overall[n_k:2*n_k]
    rmse_pearson=rmse_overall[2*n_k:3*n_k]
    mae_cosine=mae_overall[0:n_k]
    mae_msd=mae_overall[n_k:2*n_k]
    mae_pearson=mae_overall[2*n_k:3*n_k]
    params_cosine=params_overall[0:n_k]
    params_msd=params_overall[n_k:2*n_k]
    params_pearson=params_overall[2*n_k:3*n_k]
    plotLineChart("RMSE of User based KNN with ratings vs neighbors",k_overall,rmse_cosine,rmse_msd,rmse_pearson,"Number of neighbors","RMSE","Cosine","MSD","Pearson")
    plotLineChart("MAE of User based KNN with ratings vs neighbors",k_overall,mae_cosine,mae_msd,mae_pearson,"Number of neighbors","MAE","Cosine","MSD","Pearson")

    combined_dict=p.load(open('../stats/knn_means_resultscombined.p'))
    combined_dict=combined_dict.cv_results
    rmse_combined=combined_dict['RMSE']
    mae_combined=combined_dict['MAE']
    params_combined=combined_dict['params']
    k_combined=list(set(combined_dict['k']))
    k_combined.sort()
    n_k=len(k_combined)
    rmse_cosine=rmse_combined[0:n_k]
    rmse_msd=rmse_combined[n_k:2*n_k]
    rmse_pearson=rmse_combined[2*n_k:3*n_k]
    mae_cosine=mae_combined[0:n_k]
    mae_msd=mae_combined[n_k:2*n_k]
    mae_pearson=mae_combined[2*n_k:3*n_k]
    params_cosine=params_combined[0:n_k]
    params_msd=params_combined[n_k:2*n_k]
    params_pearson=params_combined[2*n_k:3*n_k]
    plotLineChart("RMSE of User based KNN with combined ratings vs neighbors",k_combined,rmse_cosine,rmse_msd,rmse_pearson,"Number of neighbors","RMSE","Cosine","MSD","Pearson")
    plotLineChart("MAE of User based KNN with combined ratings vs neighbors",k_overall,mae_cosine,mae_msd,mae_pearson,"Number of neighbors","MAE","Cosine","MSD","Pearson")
    return
    
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

# svd_plots()
knn_plots()
