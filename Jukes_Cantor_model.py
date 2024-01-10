import numpy as np
import matplotlib.pyplot as plt

def Jukes_Cantor_distance(length, observed_diff):
     p = observed_diff / length
     try:
        d = -3/4 * np.log(1-(4/3)*p) * length
     except:
         d = 0
     return d

def hamming_distance(nuc_sequence, original_nuc_seq):
    observed_diff = 0 
    for position in range(len(nuc_sequence)):
            if nuc_sequence[position] != original_nuc_seq[position]:
                observed_diff += 1
    return observed_diff


def evolve_sequence(nuc_sequence, alpha, total_substitutions):
    for nuc_position in range(len(nuc_sequence)):
        nuc = nuc_sequence[nuc_position]
        number = np.random.random() 
        if number < alpha:
            # choose one of the other 3 nucleotides
            all_bases = ['A','G','T','C'] # redefine 
            all_bases.remove(nuc)
            new_nuc = np.random.choice(all_bases)

            # update substitution to the nucleotide sequence 
            nuc_sequence = nuc_sequence[:nuc_position] + new_nuc + nuc_sequence[nuc_position+1:]

            # count the substitution
            total_substitutions += 1
    return total_substitutions, nuc_sequence


def main():
    nuc_sequence = "AGCTAGCTAGCT"
    original_nuc_seq = nuc_sequence 

    alpha = 0.01

    total_substitutions_list = []
    observed_diff_list = []
    jukes_cantor_dist_list = []

    total_substitutions = 0

    for generation in range(100):

        # evolve the sequence 
        total_substitutions, nuc_sequence = evolve_sequence(nuc_sequence, alpha, total_substitutions)
        
        # count the differences in the new sequence 
        observed_diff = hamming_distance(nuc_sequence, original_nuc_seq)

        # measure Jukes Cantor distance
        jc_dist = Jukes_Cantor_distance(len(nuc_sequence), observed_diff)

        # append to lists
        total_substitutions_list.append(total_substitutions)
        observed_diff_list.append(observed_diff)
        jukes_cantor_dist_list.append(jc_dist)


    # plot 
    plt.plot(total_substitutions_list, label = 'Substitutions that occurred')
    plt.plot(observed_diff_list, label = 'Observed differences (Hamming distance)')
    plt.plot(jukes_cantor_dist_list, label = 'Jukes Cantor distance')
    plt.legend()
    plt.show()


main()


