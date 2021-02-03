import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
import random
from pprint import pprint
#rom helperfunctions1 import determine_type_of_feature
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification
def get_potential_splits(data,random_subspace):
    potential_splits={}
    #k=75
    _, n_columns=data.shape
    column_indices=list(range(n_columns-1))
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
    return (potential_splits)

     #if random_subspace and random_subspace<=len(column_indices):
        #column_indices=random.sample(population=column_indices,k=random_subspace)
#print(potential_splits)
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]
    
    
    return data_below, data_above

    
    

def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy
def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy
def determine_best_split(data, potential_splits):
    
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

def decision_tree_algorithm(df,counter=0,min_samples=2,max_depth=5,random_subspace=None):
    
    if counter==0:
        global COLUMN_HEADERS
        COLUMN_HEADERS=df.columns
        data=df.values
    else:
        data=df
        
    if (check_purity(data))or(len(data)<min_samples)or(counter==max_depth):
        classification=classify_data(data)
        return classification
    else:
        counter+=1
        potential_splits=get_potential_splits(data,random_subspace)
        best_split_column,best_split_value=determine_best_split(data,potential_splits)
        print(best_split_column,best_split_value)
        data_below,data_above=split_data(data,best_split_column,best_split_value)
        feature_name=COLUMN_HEADERS[best_split_column]
        question="{} <= {}".format(feature_name,best_split_value)
        sub_tree={question:[]}
        yes_answer=decision_tree_algorithm(data_below,counter,min_samples,max_depth,random_subspace)
        no_answer=decision_tree_algorithm(data_above,counter,min_samples,max_depth,random_subspace)
        if yes_answer==no_answer:
            sub_tree=yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        #print(best_split_column,best_split_value)
        return sub_tree

def predict_example(example,tree):
    question=list(tree.keys())[0]
    feature_name,comparison_operator,value=question.spilit("")
    if comparision_operator=="<=":
        if example[feature_name]<=float(value):
            answer=tree[question][0]
        else:
            answer=tree[question][1]
    else:
        if str(example[feature_name]) ==value:
            answer=tree[question][0]
        else:
            answer=tree[question][1]
    if not isinstance(answer,dict):
        return answer
    else:
        residual_tree=answer
        return predict_example(example,residual_tree)
def decision_tree_predictions(test_df,tree):
    predictions=test_df.apply(predict_example, args(tree),axis=1)
    return predictions
            
