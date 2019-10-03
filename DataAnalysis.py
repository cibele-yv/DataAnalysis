# importing libraries
import pandas as pd
import numpy as np

# reading file
path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

# analyzing using data visualization
%%capture
! pip install seaborn

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

# finding correlation between colunms: 
# Write your code below and press Shift+Enter to execute 
df[['bore','stroke' ,'compression-ratio','horsepower']].corr()

# example of positive linear relationship
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

# correlation 
df[["engine-size", "price"]].corr()

# Peak-rpm is a weak linear relationship 
sns.regplot(x="peak-rpm", y="price", data=df)

# correlation = -0.1016
df[['peak-rpm','price']].corr()

# analysing categorical variable - body-style
sns.boxplot(x="body-style", y="price", data=df)

# Value Counts
# Value-counts is a good way of understanding how many units of each # characteristic/variable we have
df['drive-wheels'].value_counts()

# converting to frame
df['drive-wheels'].value_counts().to_frame()

# renaming column
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts

# renaming index
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

# BASICS OF GROUPING
df['drive-wheels'].unique()

df_group_one = df[['drive-wheels','body-style','price']]

# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one

# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot

grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot

# The heatmap plots the target variable (price) proportional to colour with respect to # the variables 'drive-wheel' and 'body-style' in the vertical and horizontal axis # respectively. This allows us to visualize how the price is related to 'drive-wheel' and # 'body-style'.
#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

# Correlation and Causation
# Correlation: a measure of the extent of interdependence between variables.
# Causation: the relationship between cause and effect between two variables.

df.corr()

from scipy import stats

# calculating pearson correlation coefficient - example not strong
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

# calculating pearson correlation coefficient - example very strong
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# ANOVA
# The Analysis of Variance (ANOVA) is a statistical method used to test whether there are # significant differences between the means of two or more groups

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)

df_gptest
grouped_test2.get_group('4wd')['price']

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   
