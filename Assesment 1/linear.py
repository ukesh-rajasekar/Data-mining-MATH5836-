## additional visualisations were made in google collab, i have also submitted my collab notebook for reference.


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random
from numpy import *  
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

#yellowbrick not working in ed
#from yellowbrick.regressor import PredictionError, ResidualsPlot 

def data_processing(df):
    
    #converting sex columns to int
    def int_conversion(sex):
        if sex == 'M':
            return 0
        elif sex == 'F':
            return 1
        else:
            return -1

    df['Sex'] = df.Sex.apply(lambda x:int_conversion(x))

    ##checking for null values
    df.isnull().values.any()

    #sorting by Rings
    df = df.sort_values(["Rings"], ascending = (True))

    #correlation matrix
    ## setting float format in 3 decimal places 
    pd.options.display.float_format = '{:,.3f}'.format
    plt.figure(figsize=(16,10))
    sns.heatmap(df.corr(), annot=True)
    plt.savefig('cov_heatmap.png')
    plt.clf()

    return df

def scatter_plot(df):

    plot_names = ['Diameter', 'Shell weight']
    for names in plot_names:
        sns.scatterplot(data = df, x = names, y = "Rings")
        plt.savefig(f'{names}.png')
        plt.clf()

def histogram_selected(df):
    feature = ['Rings', 'Shell weight', 'Diameter']
    for items in feature:
        ax = df.hist(column=items, bins=15)
        ax = ax[0]
        for x in ax:


            # Draw horizontal axis lines
            vals = x.get_yticks()
            for tick in vals:
                x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

            # Remove title
            x.set_title("")

            # Set x-axis label
            x.set_xlabel(items, labelpad=20, weight='bold', size=12)

            plt.savefig(f'histogram_{items}.png')
            plt.clf()

def get_data(df, normalise, i): 

    data_inputx = df.iloc[:,:8] # all features 
    #data_inputx = df[['Diameter', 'Shell weight']] # two features   


    if normalise == True:
        transformer = Normalizer().fit(data_inputx)  # fit does nothing.
        data_inputx = transformer.transform(data_inputx)
 
    

    data_inputy =df.iloc[:, 8] # this is target - so that last col is selected from data

    percent_test = 0.4
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=percent_test, random_state=i)

    return x_train, x_test, y_train, y_test
     
def scikit_linear_mod(x_train, x_test, y_train, y_test): 
    regr = linear_model.LinearRegression()

    ## visualizer for performance predictions not working on ed
    # visualizer = PredictionError(regr)
    # visualizer.fit(x_train, y_train)  
    # visualizer.score(x_test, y_test)  
    # visualizer.poof()

    # visualizer = ResidualsPlot(regr)
    # visualizer.fit(x_train, y_train)  
    # visualizer.score(x_test, y_test)  
    # visualizer.poof()
 

     # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
    rsquared = r2_score(y_test, y_pred) 
 
    '''residuals = y_pred - y_test
    plt.plot(residuals, linewidth=1)
 
    plt.savefig('scikit_linear.png')'''

    return rmse, rsquared, regr.coef_

 
def main(): 
    df = pd.read_csv('abalone.data')
    df.columns = ['Sex',
                'Length',
                'Diameter',
                'Height',
                'Whole weight',
                'Shucked weight',
                'Viscera weight',
                'Shell weight',
                'Rings']

    data_processing(df)

    #visualisations
    #scatter_plot(df)
    #histogram_selected(df)


    normalise = True
 
    
    max_exp = 1 # currently set to one

    rmse_list = np.zeros(max_exp)
    rsq_list = np.zeros(max_exp)

    for i in range(0,max_exp):
        
        x_train, x_test, y_train, y_test = get_data(df, normalise, i )
        rmse, rsquared, coef = scikit_linear_mod(x_train, x_test, y_train, y_test)
        
        rmse_list[i] = rmse
        rsq_list[i] = rsquared 
        

    print(rmse_list)
  
    print(rsq_list)
    
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)

    mean_rsq = np.mean(rsq_list)
    std_rsq = np.std(rsq_list)

    print(mean_rmse, std_rmse, ' mean_rmse std_rmse')

    print(mean_rsq, std_rsq, ' mean_rsq std_rsq')

if __name__ == '__main__':
     main()
