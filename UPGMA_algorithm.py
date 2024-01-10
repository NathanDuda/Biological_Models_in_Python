import re
import numpy as np


def Jukes_Cantor_distance(nuc_seq_1, nuc_seq_2, length): 
    observed_diff = 0 
    for position in range(length):  
        if nuc_seq_1[position] != nuc_seq_2[position]:
            observed_diff += 1
    p = observed_diff / length
    if 0 < p < 3/4:
        d = -(3/4) * np.log(1 - (4/3) * p) * length  
    else:
        d = 0
    return d


def calc_init_distance_matrix(sequences):
    num_seq = len(sequences)
    d_mat = np.zeros((num_seq, num_seq))
    for i in range(num_seq):
        for j in range(num_seq):
            if i < j: # calculate only top half of matrix 
                d_mat[i][j] = Jukes_Cantor_distance(sequences[i], sequences[j], len(sequences[i]))
    return d_mat


def shortest_dist(d_mat):
    i,j = np.unravel_index(np.argmin(d_mat + np.diag([np.inf]*d_mat.shape[0])), d_mat.shape)
    return i, j


def update_d_mat(d_mat, i, j):
    combined_row = (d_mat[i] + d_mat[j]) / 2
    combined_col = (d_mat[:, i] + d_mat[:, j]) / 2

    d_mat = np.insert(d_mat, d_mat.shape[0], combined_row, axis=0)
    d_mat = np.insert(d_mat, d_mat.shape[1], np.append(combined_col, 0), axis=1)

    d_mat = np.delete(d_mat, [i, j], axis=0) 
    d_mat = np.delete(d_mat, [i, j], axis=1)

    return d_mat


def upgma(sequences): # main
    d_mat = calc_init_distance_matrix(sequences)
    clusters = [(i,) for i in range(len(sequences))]

    while len(clusters) > 1:
        i, j = shortest_dist(d_mat)
        d_mat = update_d_mat(d_mat, i, j)

        clusters[i] = (clusters[i],clusters[j])   # add j to the i cluster 
        clusters.pop(j)                           # remove the lone j cluster 

    newick = str(clusters[0])
    newick = re.sub(r'\((\d+),\)', r'\1', newick) # remove the extra parentheses and commas around the numbers with regex
    return newick


sequences = ['ATCG', 'ATTG', 'TTTG', 'ATCC', 'TTCG']
newick = upgma(sequences)

print(newick)  

