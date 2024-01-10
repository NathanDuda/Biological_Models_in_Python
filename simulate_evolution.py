import numpy as np

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


def mutate(sequence, mutation_rate):
    nucleotides = ['A','C','G','T']
    new_sequence = ''
    for nucleotide in sequence:
        if np.random.random() < mutation_rate:
            new_sequence += np.random.choice([nuc for nuc in nucleotides if nuc != nucleotide])
        else:
            new_sequence += nucleotide
    return new_sequence


def simulate_evolution(node, depth, mutation_rate, max_depth):
    # check if should terminate or not 
    if depth < max_depth:
        num_children = np.random.choice([1,2],p=[0.8,0.2])
        for i in range(num_children):
            mutated_seq = mutate(node.sequence, mutation_rate)
            child_node = TreeNode(f"{node.name}.{i+1}", mutated_seq)
            node.add_child(child_node)
            simulate_evolution(child_node, depth + 1, mutation_rate, max_depth)


def draw_tree(node, far):
    indent = " " * far
    print(indent,far)
    for child in node.children:
        draw_tree(child, far + 1)


# usage
root_sequence = 'ATGCATGCATGC'
mutation_rate = 0.1
root = TreeNode("S",root_sequence)

depth = 0 
max_depth = 8 
node = root 
simulate_evolution(node, depth, mutation_rate, max_depth)


# view tree
draw_tree(node,0)

# print newick tree
newick_tree = root.newick()
print(newick_tree)
