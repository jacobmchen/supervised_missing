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
        for learning_method in learning_methods:
            subset_data = data[data['forest'] == learning_method]
            
            method_names = set(subset_data['method'])
            
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

            table_methods = []
            coef = []
            pval = []
            for i in range(len(method_names)):
                table_methods.append(results.params.index[i+1].removeprefix('method_'))
                coef.append(results.params[i+1])
                pval.append(results.pvalues[i+1])

            df = pd.DataFrame({'Method Name': table_methods, 'Coefficient': coef, 'P-values': pval})

            fix, ax = plt.subplots()

            # Hide the axes
            ax = plt.gca()
            ax.axis('off')
            ax.set_title(learning_method)

            table = pd.plotting.table(ax, df, loc='center', cellLoc='center', colWidths=list([.2]*len(df)))
            pdf.savefig()
            plt.close()