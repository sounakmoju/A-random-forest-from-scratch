import numpy as np
import pandas as pd
import random
from pprint import pprint
from pprint import pprint
from dec_tree  import decision_tree_algorithm,decision_tree_predictions
from help_func import train_test_split,calculate_accuracy
import matplotlib.pyplot as plt
#regression_tree_2
df=pd.read_csv("winequality-red.csv")
df["label"]=df.quality
df=df.drop("quality",axis=1)
#df=df.drop("quality",axis=1)
#column_names=[]
#for column in df.columns:
    #name=column.replace("","_")
    #column_names.append(name)
#df.columns=column_names
#df=df.rename(columns={"quality":"label"})
print(df.head())
##print(df.fixedacidity())
wine_quality=df.label.value_counts(normalize=True)
wine_quality=wine_quality.sort_index()
wine_quality.plot(kind="bar")
#plt.show()
def transform_label(value):
    if value<=5:
        return "bad"
    else:
        return "good"
df["label"]=df.label.apply(transform_label)
wine_quality=df.label.value_counts(normalize=True)
wine_quality[["bad","good"]].plot(kind="bar")
print(wine_quality)
random.seed(0)
train_df,train_df=train_test_split(df,test_size=0.2)
def get_potential_splits(data,random_subspace):
    potential_splits={}
    #k=75
    _, n_columns=data.shape
    column_indices=list(range((n_columns-1)))
    if random_subspace and random_subspace<=len(column_indices):
        column_indices=random.sample(population=column_indices,k=random_subspace)


    for column_index in column_indices:
        potential_splits[column_index]=[]
        values=data[:,column_index]
        values1=sorted(values)
        k=round(((len(values1))-1)/2)
        min_value=min(values1)
        max_value=max(values1)
        #print(min_value)
        for i in range(k):
            #print(i)
            potential_split=min_value+(i*((max_value-min_value)/(k+1)))
            potential_splits[column_index].append(potential_split)
            #print(potential_splits)
            
            
        #print(values1)
    return potential_splits
    
   

def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices=np.random.randint(low=0,high=len(train_df),size=n_bootstrap)
    df_bootstrapped=train_df.iloc[bootstrap_indices]

    return df_bootstrapped
def random_forest_algorithm(train_df,n_trees,n_bootstrap,n_features,dt_max_depth):
    forest=[]
    for i in range(n_trees):
        df_bootstrapped=bootstrapping(train_df,n_bootstrap)
        tree=decision_tree_algorithm(df_bootstrapped,max_depth=dt_max_depth,random_subspace=n_features)
        forest.append(tree)
    return forest
forest=random_forest_algorithm(train_df,n_trees=4,n_bootstrap=800,n_features=4,dt_max_depth=4)
print(forest[3])
