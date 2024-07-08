import scipy.stats as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter


# generates n income data points for parents and children
# returns two arrays, child incomes and parent incomes
def generate_incomes(n, pj):
    ln_y_parent = st.norm(0, 1).rvs(size=n)
    residues = st.norm(0, 1).rvs(size=n)
    return np.exp(pj * ln_y_parent + residues),  np.exp(ln_y_parent)

# returns a pd.series where each income value from l is replaced by its
# quantile number from q_dict
def quantiles(l, nb_quantiles):
    size = len(l)
    l_sorted = l.copy()
    l_sorted = l_sorted.sort_values()
    quantiles = np.round(np.arange(1, nb_quantiles + 1, nb_quantiles / size) - 0.5 + 1. / size)
    q_dict = {a: int(b) for a, b in zip(l_sorted, quantiles)}
    return pd.Series([q_dict[e] for e in l])

# convet child and parent income arrays to pandas series
# converts the income and quantile data into a sample
def compute_quantiles(y_child, y_parents, nb_quantiles):
    y_child = pd.Series(y_child)
    y_parents = pd.Series(y_parents)
    c_i_child = quantiles(y_child, nb_quantiles)
    c_i_parent = quantiles(y_parents, nb_quantiles)
    sample = pd.concat([y_child, y_parents, c_i_child, c_i_parent], axis=1)
    sample.columns = ["y_child", "y_parents", "c_i_child", "c_i_parent"]
    return sample

# computes the distribution of quantile counts for a given parent quantile
# counts is a dataFrame containing quantile counts
# total is the sum of all counts
# Iterates over each parent quantile, calculates the proportion of child quantiles
# within each parent quantile and returns the distribution
def distribution(counts, nb_quantiles):
    distrib = []
    total = counts["counts"].sum()
    if total == 0:
        return [0] * nb_quantiles
    for q_p in range(1, nb_quantiles + 1):
        subset = counts[counts.c_i_parent == q_p]
        if len(subset):
            nb = subset["counts"].values[0]
            distrib.append(nb / total)
        else:
            distrib.append(0)
    return distrib

# groups the sample dataframe by child adn parent quantiles and counts occurences
# consutructs matrix where each row represents the distrubution of 
# parent quantiles for a given child quantile
def conditional_distributions(sample, nb_quantiles):
    counts = sample.groupby(["c_i_child", "c_i_parent"]).size().reset_index(name='counts')
    mat = []
    for child_quantile in np.arange(nb_quantiles) + 1:
        subset = counts[counts.c_i_child == child_quantile]
        mat.append(distribution(subset, nb_quantiles))
    return np.array(mat)

# plot
def plot_conditional_distributions(p, cd, nb_quantiles):
    plt.figure()
    cumul = np.array([0] * nb_quantiles)
    for i, child_quantile in enumerate(cd):
        plt.bar(np.arange(nb_quantiles) + 1, child_quantile, bottom=cumul, width=0.95, label=str(i + 1) + "e")
        cumul += np.array(child_quantile)
    plt.axis([0.5, nb_quantiles * 1.3, 0, 1])
    plt.title("p=" + str(p))
    plt.legend()
    plt.xlabel("c_i_parent")
    plt.ylabel("probability of having c_i_child")
    plt.show()

# retrieves the conditional probability from the matrix mat given 
# specific parent and child quantiles
# returns the probability of a child being in 
# c_i_child given the parent is in c_i_parent 
def proba_cond(c_i_parent, c_i_child, mat):
    return mat[c_i_child, c_i_parent]