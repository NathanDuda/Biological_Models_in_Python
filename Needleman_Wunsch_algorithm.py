import numpy as np


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


def main():
    seq_1 = 'CGATAGTCTATACGTTTTATCACCAGTTAGTCAGACTACGATACTATGCATGCA'
    seq_2 = 'CGATAGTCTATACCACCAGTTAGTCAGACTACGATACTAATGCA'

    match_point = 1
    mismatch_penalty = 0
    gap_penalty = -3

    score_matrix, backtrack_matrix = initialize_matrices(len(seq_1) + 1, len(seq_2) + 1, gap_penalty)

    backtrack_matrix = get_backtrack_matrix(seq_1, seq_2, score_matrix, backtrack_matrix, match_point, mismatch_penalty, gap_penalty)

    aligned_seq_1, aligned_seq_2 = needleman_wunsch(seq_1, seq_2, backtrack_matrix)
    
    print(aligned_seq_1)
    print(aligned_seq_2)


main()

