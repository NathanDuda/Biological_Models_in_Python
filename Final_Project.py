import numpy as np 
import re
import os
from Bio import Phylo

os.chdir('C:/Users/17735/Downloads/BIOL5514/')


def read_seq_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sequences_names = [line[1:] for line in lines if line.startswith('>')]
    sequences = [line for line in lines if not line.startswith('>')]
    return sequences, sequences_names


def JC_plus_indels(sequence, indel_mutation_rate, sub_mutation_rate):
    nucleotides = ['A', 'C', 'G', 'T']
    new_sequence = ''
    for nucleotide in sequence:
        if np.random.random() < sub_mutation_rate:  # if substitution
            new_sequence += np.random.choice([nuc for nuc in nucleotides if nuc != nucleotide])
        else:
            new_sequence += nucleotide
        if np.random.random() < indel_mutation_rate:  # if indel
            if np.random.random() < 0.5:  # if insertion (50%) # otherwise, deletion (nothing added)
                new_sequence += nucleotide
                new_sequence += np.random.choice([nuc for nuc in nucleotides if nuc != nucleotide])
    return new_sequence


class TreeNode:
    def __init__(self,name,sequence):
        self.name = name
        self.sequence = sequence
        self.children = []
    def add_child(self, child):
        self.children.append(child)
    def newick(self):
        if not self.children:
            return self.name
        return "(" + ",".join(child.newick() for child in self.children) + ")" + self.name


def simulate_evolution(node, depth, indel_mutation_rate, sub_mutation_rate, max_depth):
    if depth < max_depth:
        num_children = np.random.choice([1,2],p=[0.8,0.2])
        for i in range(num_children):
            mutated_seq = JC_plus_indels(node.sequence, indel_mutation_rate, sub_mutation_rate)
            child_node = TreeNode(f"{node.name}.{i}", sequence=mutated_seq)
            node.add_child(child_node)
            simulate_evolution(child_node, depth + 1, indel_mutation_rate, sub_mutation_rate, max_depth)


def get_leaf_sequences(node, sequences, sequences_names):
    if node.children != []:
        for child in node.children:
            get_leaf_sequences(child, sequences, sequences_names)    
    if node.children == []:
        sequences.append(node.sequence)
        sequences_names.append(node.name)
    return sequences, sequences_names 


def write_fasta_file(file_path, sequences, sequences_names):
    with open(file_path, 'w') as file:
        for name, sequence in zip(sequences_names, sequences):
            file.write(f'>{name}\n{sequence}\n')

###########################################################################################################################################
# SECOND PROGRAM
###########################################################################################################################################

def initialize_matrices(rows, cols, gap_penalty):
    score_matrix = np.zeros((rows, cols), dtype=int)
    backtrack_matrix = np.zeros((rows, cols), dtype=object) # allow it to hold 'diag','up','down'

    score_matrix[0, 0] = 0
    score_matrix[0, 1:] = np.arange(1, cols) * gap_penalty
    score_matrix[1:, 0] = np.arange(1, rows) * gap_penalty

    backtrack_matrix[0, 0] = 'diag'
    backtrack_matrix[0, 1:] = "left"
    backtrack_matrix[1:, 0] = "up"

    return score_matrix, backtrack_matrix


def get_backtrack_matrix(seq_1, seq_2, score_matrix, backtrack_matrix, match_point, mismatch_penalty, gap_penalty):
    for i in range(1, len(seq_1) + 1):
        for j in range(1, len(seq_2) + 1):
            match_score = score_matrix[i-1, j-1] + (match_point if seq_1[i-1] == seq_2[j-1] else mismatch_penalty)
            gap_left_score = score_matrix[i, j-1] + gap_penalty
            gap_up_score = score_matrix[i-1, j] + gap_penalty

            scores = [match_score, gap_left_score, gap_up_score]
            score_matrix[i, j] = max(scores) 

            if max(scores) == match_score == scores[0]:
                backtrack_matrix[i, j] = "diag"
            elif max(scores) == gap_up_score == scores[2]:
                backtrack_matrix[i, j] = "up"
            elif max(scores) == gap_left_score == scores[1]:
                backtrack_matrix[i, j] = "left"

    return backtrack_matrix


def needleman_wunsch(seq_1, seq_2, backtrack_matrix):
    aligned_seq_1 = '' 
    aligned_seq_2 = ''

    traceback_row = len(seq_1)
    traceback_col = len(seq_2)

    while traceback_row > 0 or traceback_col > 0:
        if traceback_row > 0 and traceback_col > 0 and backtrack_matrix[traceback_row, traceback_col] == "diag":
            aligned_seq_1 += seq_1[traceback_row-1]
            aligned_seq_2 += seq_2[traceback_col-1]
            traceback_row -= 1
            traceback_col -= 1
        elif traceback_row > 0 and (traceback_col == 0 or backtrack_matrix[traceback_row, traceback_col] == "up"):
            aligned_seq_1 += seq_1[traceback_row-1]
            aligned_seq_2 += '-'
            traceback_row -= 1
        elif traceback_col > 0 and (traceback_row == 0 or backtrack_matrix[traceback_row, traceback_col] == "left"):
            aligned_seq_1 += '-'
            aligned_seq_2 += seq_2[traceback_col-1]
            traceback_col -= 1

    aligned_seq_1 = aligned_seq_1[::-1] # reverse 
    aligned_seq_2 = aligned_seq_2[::-1]

    return aligned_seq_1, aligned_seq_2


def align_main(seq_1,seq_2):
    match_point = 1
    mismatch_penalty = 0
    gap_penalty = -3

    score_matrix, backtrack_matrix = initialize_matrices(len(seq_1) + 1, len(seq_2) + 1, gap_penalty)
    backtrack_matrix = get_backtrack_matrix(seq_1, seq_2, score_matrix, backtrack_matrix, match_point, mismatch_penalty, gap_penalty)

    aligned_seq_1, aligned_seq_2 = needleman_wunsch(seq_1, seq_2, backtrack_matrix)
    
    return aligned_seq_1, aligned_seq_2


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
                sequences[i], sequences[j] = align_main(sequences[i], sequences[j])
                d_mat[i][j] = Jukes_Cantor_distance(sequences[i], sequences[j], len(sequences[i]))
    return d_mat, sequences


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


def upgma(sequences, sequences_names):  
    d_mat, aligned_sequences = calc_init_distance_matrix(sequences)
    clusters = [(i,) for i in range(len(sequences))]
    cluster_names = list(sequences_names)

    while len(clusters) > 1:
        i, j = shortest_dist(d_mat)
        d_mat = update_d_mat(d_mat, i, j)

        new_cluster = (clusters[i], clusters[j])
        new_name = f'({cluster_names[i]}, {cluster_names[j]})'
        
        clusters[i] = new_cluster
        cluster_names[i] = new_name
        
        clusters.pop(j)
        cluster_names.pop(j)

    newick = str(cluster_names[0])
    newick = re.sub(r'\((\d+),\)', r'\1', newick)
    return newick, aligned_sequences

###########################################################################################################################################
# MAIN
###########################################################################################################################################

def MAIN():
    sub_mutation_rate = 0.1
    indel_mutation_rate = 0.05
    max_depth = 12 # generations 

    original_seq, orig_seq_name = read_seq_file('./original_nuc_sequence.txt')

    root = TreeNode("S",original_seq)
    depth = 0 
    simulate_evolution(root, depth, indel_mutation_rate, sub_mutation_rate, max_depth)

    newick = root.newick()
    with open('./newick_tree.txt', "w") as output_file:
        output_file.write(newick)
    print('\n')
    print('Part 1 Newick:')
    Phylo.draw_ascii(Phylo.read('./newick_tree.txt', "newick")) # gives error if linear tree (no branching) 
    
    sequences, sequences_names = get_leaf_sequences(root, [], []) # empty sequences and sequence names 

    write_fasta_file('./simulated_sequences.txt', sequences, sequences_names)


    # Part 2 
    sequences, sequences_names = read_seq_file('./simulated_sequences.txt')

    newick_P2, aligned_sequences = upgma(sequences, sequences_names)
    with open('./newick_tree_P2.txt', "w") as output_file:
        output_file.write(newick_P2)
    print('\n')
    print('Part 2 Newick:')
    Phylo.draw_ascii(Phylo.read('./newick_tree_P2.txt', "newick")) # gives error if linear tree (no branching) 

MAIN()

