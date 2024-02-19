import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_coefficient_table(data, filename):

    learning_methods = set(data['forest'])

    with PdfPages(filename) as pdf:
        results_df = pd.DataFrame()
        subset_data = data[data['forest'] == 'DECISION TREE']

        # get the full list of imputation methods for the full table
        full_table_methods = list(set(subset_data['method']))
        full_table_methods = sorted(full_table_methods)

        full_table_methods.append('const')

        results_df['Imputation Method'] = full_table_methods
        for learning_method in learning_methods:
            subset_data = data[data['forest'] == learning_method]
            
            method_names = list(set(subset_data['method']))
            
            # create categorical dummy variables
            dummied_data = pd.get_dummies(subset_data, columns=['method', 'fold'], drop_first=False)
            # drop the columns of data that we don't want to include in our regression
            dummied_data = dummied_data.drop(columns=['index', 'score', 'forest', 'R2'])

            dummied_data = sm.add_constant(dummied_data)

            # get the target variable
            y = subset_data['R2']

            # Fit the GLM (linear model)
            model = sm.OLS(y, dummied_data)
            results = model.fit()

            # not every learning algorithm has every imputation method, so make
            # full length arrays for coef and pval values and populate it with
            # na values
            coef = [np.nan for _ in range(len(full_table_methods))]
            pval = [np.nan for _ in range(len(full_table_methods))]

            for i in range(len(method_names)+1):
                # get the current method_name that we're working with
                method_name = results.params.index[i].removeprefix('method_')
                try:
                    # get the index in the full table for the imputation method we are working
                    # with right now
                    index = full_table_methods.index(method_name)
                    
                    coef[index] = round(results.params[i], 2)
                    pval[index] = round(results.pvalues[i], 2)
                except ValueError:
                    pass

            # add the coefficient and p-values to the final table dataframe
            results_df['Coef_'+learning_method] = coef
            results_df['Pval_'+learning_method] = pval

        fix, ax = plt.subplots()

        # Hide the axes
        ax = plt.gca()
        ax.axis('off')

        table = pd.plotting.table(ax, results_df, loc='center', cellLoc='center', colWidths=list([.1]*len(results_df)))

        # code for setting font size
        # table.auto_set_font_size(False)
        # table.set_fontsize(4)

        pdf.savefig()
        plt.close()