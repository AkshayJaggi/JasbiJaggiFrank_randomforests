# All Imports
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve,GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import preprocessing
import inspect
import pandas as pd
import graphviz
import numpy as np
from matplotlib import rcParams, pyplot as plt
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from scipy import stats

# Get Max Estimator
# Takes: The list of features, the list of labels, the max allowable depth, the minimum gini decrease
# Does:  High level: finds the best example tree in a forest
#        Splits the data into training and test splits
#        For each split, train a random forest on this split
#        For all the trees in the forest, select the tree with the best performance on the test data
#        Finally, train a final random forest on all the data and report its performance
# Gives: The performance of the forest, the best tree
def get_max_estimator(features,labels,depth,mind):
    st = 10
    ss = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=st)
    clf = rfc(n_estimators=10,max_depth=depth,max_features=None,min_impurity_decrease=mind,random_state=st)
    max_tree = 0
    for train, test in cv.split(features,labels):
        x_train, x_test, y_train, y_test = 
        features.iloc[train], features.iloc[test],labels.iloc[train],labels.iloc[test]
        clf.fit(x_train,y_train)
        for estimator in clf.estimators_:
            if max_tree is 0 or estimator.score(x_test,y_test) > max_tree.score(x_test,y_test):
                max_tree = estimator
    scores = cross_val_score(clf, features, labels, cv=ss)
    return scores, max_tree

# Plot Max Estimator
# Takes: An example tree, the list of features, the feature classes, the plot name
# Does:  Plots the best tree as a figure, saves as a file
# Gives: Nothing
def plot_max_estimator(max_tree,features,classes, name):
    dot_data = tree.export_graphviz(max_tree, out_file=None,
                                        feature_names=list(features),
                                        class_names=classes,
                                        filled=True, rounded=True,
                                        special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('dissertation_plots/'+name)
    graph
    
# Tradeoff
# Takes: The list of features and labels, the max depth, min impurity decrease list
# Does:  High level: tests the effect of changing the min impurity decrease
# Gives: List of all performance scores and all best trees with many different ginis
def tradeoff(features, labels,depth,minds):
    all_scores = []
    all_trees  = []
    for i in np.nditer(minds):
        scores, mt = get_max_estimator(features,labels,depth,i)
        all_scores.append(scores)
        all_trees.append(mt)
    return all_scores, all_trees


# Tradeoff
# Takes: all the gini decrease values, counts, the mean performance, the std of the performance
# Does:  Plotting the tradeoff between the gini decrease and model performance
# Gives: a plot!
def plot_tradeoff(minds,counts,means,stds):
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(minds, counts, 'b-')
    axs[0,0].set_xlabel('min impurity decrease')
    axs[0,0].set_ylabel('Max Tree Depth', color='b')
    axs[0,0].tick_params('y', colors='b')
    ax2 = axs[0,0].twinx()
    ax2.plot(minds, stds, 'r-')
    ax2.set_ylabel('Std Dev', color='r')
    ax2.tick_params('y', colors='r')
    axs[1,0].plot(minds, counts, 'b-')
    axs[1,0].set_xlabel('min impurity decrease')
    axs[1,0].set_ylabel('Max Tree Depth', color='b')
    axs[1,0].tick_params('y', colors='b')
    ax2 = axs[1,0].twinx()
    ax2.plot(minds, means, 'g-')
    ax2.set_ylabel('Accuracy', color='g')
    ax2.tick_params('y', colors='g')
    axs[0,1].plot(minds, means, 'g-')
    axs[0,1].set_xlabel('min impurity decrease')
    axs[0,1].set_ylabel('Accuracy', color='g')
    axs[0,1].tick_params('y', colors='g')
    ax2 = axs[0,1].twinx()
    ax2.plot(minds, stds, 'r-')
    ax2.set_ylabel('Std Dev', color='r')
    ax2.tick_params('y', colors='r')
    plt.tight_layout()
    plt.show()
    
# Important Features
# Takes: all features and an example tree
# Does:  Collecting all nonzero importance features
# Gives: a list of nonzero importance features
def important_features(features,input_tree):
    nonzero = []
    for row in zip(features,input_tree.feature_importances_):
        if row[1]:
            nonzero.append(row)
    return nonzero 

# Make the labels binary and flip them if necessary
def make_binary_labels(labels, flip):
    lb = preprocessing.LabelBinarizer()
    lb.fit(labels)
    blabels = lb.transform(labels)
    if flip:
        i = 0
        for label in blabels:
            blabels[i] = not label
            i += 1
    return blabels

# random forest classifier learning curve
# Takes: array of the features, array of the labels (EX vs IN),
#        training sizes of the experiments, gini threshold,
#        how to score the model, how to collect error bands, what is considered positive, 
#        whether or not to return the raw or summarized test scores
# Does:  Set Random Seed
#        Binarize the Labels
#        Flip this binarization if we want to look at the "other direction"
#        Define the random forest classifer 
#        Define the data splitting for training/testing
#        Run the learning curve (train model on increasingly large subsets of the data)
#        Report Results
# Gives: The training sizes, the model performances
def rfc_learning_curve(features,labels,
                       training_sizes,
                       gini,
                       score='accuracy',perc=False,return_raw=False):
    st = 10
    clf = rfc(n_estimators=20,max_depth=8,max_features=None,min_impurity_decrease=gini,random_state=st)
    ss = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=st)
    train_sizes, train_scores, test_scores = learning_curve(
        clf, features, labels, cv=ss, train_sizes=training_sizes,shuffle=True,scoring=score,random_state=st)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_var  = np.percentile(test_scores,95,axis=1) if perc else np.std(test_scores,axis=1)
    if return_raw:
        return train_sizes,test_scores 
    else:
        return train_sizes,test_scores_mean, test_scores_var

# binary random forest classifier learning curve
# Takes: array of the features, array of the labels (EX vs IN),
#        training sizes of the experiments, gini threshold,
#        how to score the model, how to collect error bands, what is considered positive, 
#        whether or not to return the raw or summarized test scores
# Does:  Set Random Seed
#        Binarize the Labels
#        Flip this binarization if we want to look at the "other direction"
#        Define the random forest classifer 
#        Define the data splitting for training/testing
#        Run the learning curve (train model on increasingly large subsets of the data)
#        Report Results
# Gives: The training sizes, the model performances
def rfc_learning_curve_bin(features,labels,
                           training_size,
                           gini,
                           score='accuracy', perc=False,flip=False,return_raw=False):
    
    random_seed = 10
    blabels = make_binary_labels(labels, flip)
    clf = rfc(n_estimators=20,max_depth=8,max_features=None,min_impurity_decrease=gini,random_state=st)
    ss = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=st)
    train_sizes, train_scores, test_scores = 
    learning_curve(clf, features, blabels.flatten(), cv=ss,
                   train_sizes=training_size,shuffle=True,scoring=score,random_state=st)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_var  = np.percentile(test_scores,95,axis=1) if perc else np.std(test_scores,axis=1)
    if return_raw:
        return train_sizes,test_scores 
    else:
        return train_sizes,test_scores_mean, test_scores_var 
    
# Plot Learning Curve
def plot_learning_curves(train_sizes, mean_std_array,title, xlabel,labels):
    colors = ['#00BFC4','#F8766D']
    i = 0
    for mean_std in mean_std_array:
        mean = mean_std[0]
        std = mean_std[1]
        plt.fill_between(train_sizes, mean - std, mean + std, alpha=0.1,color=colors[i%2])
        plt.plot(train_sizes, mean,color=colors[i%2],label=labels[i])
        i+=1
    plt.xlabel('Number of Training Examples')
    plt.ylabel(xlabel)
    plt.title(title)
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.show()