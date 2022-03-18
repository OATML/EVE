import functools
import os
import time
from collections import defaultdict

import numba
import numpy as np
import pandas as pd
import torch
from numba import prange
from tqdm import tqdm
import multiprocessing

# constants
GAP = "-"
MATCH_GAP = GAP
INSERT_GAP = "."

ALPHABET_PROTEIN_NOGAP = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_PROTEIN_GAP = GAP + ALPHABET_PROTEIN_NOGAP


class MSA_processing:
    def __init__(self,
                 MSA_location="",
                 theta=0.2,
                 use_weights=True,
                 weights_location="./data/weights",
                 preprocess_MSA=True,
                 threshold_sequence_frac_gaps=0.5,
                 threshold_focus_cols_frac_gaps=0.3,
                 remove_sequences_with_indeterminate_AA_in_focus_cols=True,
                 num_cpus=1,
                 ):

        """
        Parameters:
        - msa_location: (path) Location of the MSA data. Constraints on input MSA format:
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corresponding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that;
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path) File to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        - num_cpus: (int) Number of CPUs to use for parallel weights calculation processing
        """
        np.random.seed(2021)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = ALPHABET_PROTEIN_NOGAP
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols

        # Defined by gen_alignment
        self.aa_dict = {}
        self.focus_seq_name = ""
        self.seq_name_to_sequence = defaultdict(str)
        self.focus_seq, self.focus_cols, self.focus_seq_trimmed, self.seq_len, self.alphabet_size = [None] * 5
        self.focus_start_loc, self.focus_stop_loc = None, None
        self.uniprot_focus_col_to_wt_aa_dict, self.uniprot_focus_col_to_focus_idx = None, None
        self.one_hot_encoding, self.weights, self.Neff, self.num_sequences = [None] * 4

        # Defined by create_all_singles
        self.mutant_to_letter_pos_idx_focus_list = None
        self.all_single_mutations = None

        # Fill in the instance variables
        self.gen_alignment(num_cpus=num_cpus)
        self.create_all_singles()

    def gen_alignment(self, num_cpus=1):
        """ Read training alignment and store basics in class instance """
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    if i == 0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line
        print("Number of sequences in MSA (before preprocessing):", len(self.seq_name_to_sequence))

        ## MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence'])
            # Data clean up
            msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".", "-")).apply(
                lambda x: ''.join([aa.upper() for aa in x]))
            # Remove columns that would be gaps in the wild type
            non_gap_wt_cols = [aa != '-' for aa in msa_df.sequence[self.focus_seq_name]]
            msa_df['sequence'] = msa_df['sequence'].apply(
                lambda x: ''.join([aa for aa, non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind]))
            assert 0.0 <= self.threshold_sequence_frac_gaps <= 1.0, "Invalid fragment filtering parameter"
            assert 0.0 <= self.threshold_focus_cols_frac_gaps <= 1.0, "Invalid focus position filtering parameter"
            msa_array = np.array([list(seq) for seq in msa_df.sequence])
            gaps_array = np.array(list(map(lambda seq: [aa == '-' for aa in seq], msa_array)))
            # Identify fragments with too many gaps
            seq_gaps_frac = gaps_array.mean(axis=1)
            seq_below_threshold = seq_gaps_frac <= self.threshold_sequence_frac_gaps
            print("Proportion of sequences dropped due to fraction of gaps: " + str(
                round(float(1 - seq_below_threshold.sum() / seq_below_threshold.shape) * 100, 2)) + "%")
            # Identify focus columns
            columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
            index_cols_below_threshold = columns_gaps_frac <= self.threshold_focus_cols_frac_gaps
            print("Proportion of non-focus columns removed: " + str(
                round(float(1 - index_cols_below_threshold.sum() / index_cols_below_threshold.shape) * 100, 2)) + "%")
            # Lower case non focus cols and filter fragment sequences
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join(
                [aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in
                 zip(x, index_cols_below_threshold)]))
            msa_df = msa_df[seq_below_threshold]
            # Overwrite seq_name_to_sequence with clean version
            self.seq_name_to_sequence = defaultdict(str)
            for seq_idx in range(len(msa_df['sequence'])):
                self.seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.sequence[seq_idx]

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s != '-']
        self.focus_seq_trimmed = "".join([self.focus_seq[ix] for ix in self.focus_cols])
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        focus_loc = self.focus_seq_name.split("/")[-1]
        start, stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col + int(start): self.focus_seq[idx_col] for idx_col in self.focus_cols}
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col + int(start): idx_col for idx_col in self.focus_cols}

        # Move all letters to CAPS; keeps focus columns only
        for seq_name, sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".", "-")
            self.seq_name_to_sequence[seq_name] = "".join(
                [sequence[ix].upper() for ix in self.focus_cols])  # Makes a List[str] instead of str

        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name, sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                del self.seq_name_to_sequence[seq_name]

        # Encode the sequences
        print("One-hot encoding sequences")
        self.one_hot_encoding = one_hot_3D(
            seq_keys=self.seq_name_to_sequence.keys(),  # Note: Dicts are unordered for python < 3.6
            seq_name_to_sequence=self.seq_name_to_sequence,
            msa_data=self,  # Simply pass ourselves as the msa_data object
        )

        # TODO(Lood) refactor this out so that it can be used in isolation
        if self.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                print("Loaded sequence weights from disk")
            except:
                print("Computing sequence weights")
                # EVCouplings weights calc:
                alphabet_mapper = map_from_alphabet(ALPHABET_PROTEIN_GAP, default=GAP)
                arrays = []
                for seq in self.seq_name_to_sequence.values():
                    arrays.append(np.array(list(seq)))
                sequences = np.vstack(arrays)
                sequences_mapped = map_matrix(sequences, alphabet_mapper)
                print("Compiling JIT function")
                start = time.perf_counter()
                # Compile jit function
                _ = calc_weights_evcouplings(sequences_mapped[:10], identity_threshold=1 - self.theta,
                                             empty_value=0, num_cpus=num_cpus)  # GAP = 0
                print("JIT function compiled/run in {} seconds".format(time.perf_counter() - start))
                # TODO temporary speed tests
                # del sequences
                print("Checking runtime for JIT function with different args")
                start = time.perf_counter()
                _ = calc_weights_evcouplings(sequences_mapped[:11], identity_threshold=1 - self.theta,
                                             empty_value=0, num_cpus=num_cpus)  # GAP = 0
                print("JIT function with length 11 run in {} seconds".format(time.perf_counter() - start))

                print("100 seqs:")
                start = time.perf_counter()
                _ = calc_weights_evcouplings(sequences_mapped[:100], identity_threshold=1 - self.theta,
                                             empty_value=0, num_cpus=num_cpus)  # GAP = 0
                print("JIT function with length 100 run in {} seconds".format(time.perf_counter() - start))
                ###########################################################
                print("1000 seqs:")
                start = time.perf_counter()
                _ = calc_weights_evcouplings(sequences_mapped[:1000], identity_threshold=1 - self.theta,
                                             empty_value=0, num_cpus=num_cpus)  # GAP = 0
                print("JIT function with length 1000 run in {} seconds".format(time.perf_counter() - start))
                #######################################################################################################
                print("10k seqs:")
                start = time.perf_counter()
                _ = calc_weights_evcouplings(sequences_mapped[:10000], identity_threshold=1 - self.theta,
                                             empty_value=0, num_cpus=num_cpus)  # GAP = 0
                print("JIT function with length 10k run in {} seconds".format(time.perf_counter() - start))
                ##################################
                print("100k seqs:")
                start = time.perf_counter()
                _ = calc_weights_evcouplings(sequences_mapped[:100000], identity_threshold=1 - self.theta,
                                             empty_value=0, num_cpus=num_cpus)  # GAP = 0
                print("JIT function with length 100k run in {} seconds".format(time.perf_counter() - start))

                print("Starting EVCouplings calculation")
                start = time.perf_counter()
                ev = calc_weights_evcouplings(sequences_mapped, identity_threshold=1 - self.theta,
                                              empty_value=0, num_cpus=num_cpus)  # GAP = 0
                end = time.perf_counter()
                print(f"EVCouplings weights took {end - start:.2f} seconds")
                # EVE weights calc:
                list_seq = self.one_hot_encoding.numpy()
                start = time.perf_counter()
                # TODO check memory usage for multiprocessing
                eve = compute_sequence_weights(list_seq, self.theta, num_cpus=num_cpus)
                end = time.perf_counter()
                print(f"EVE weights took {end - start:.2f} seconds")
                # tmp check diffs
                for old, new, seq, idx in zip(eve[eve != ev], ev[eve != ev], sequences[eve != ev],
                                              np.where(eve != ev)[0]):
                    print(f"Sequence {idx} {''.join(seq)}: EVE {1 / old} -> ev {1 / new}")

                self.weights = ev
                # del sequences_mapped
                print("Saving sequence weights to disk")
                # Also a temporary check
                assert np.array_equal(eve, ev), f"EVCouplings and EVE weights are not equal. EVcouplings weights: {ev}, EVE weights: {eve}"
                print("EVCouple and EVE weights are equal")  #tmp
                np.save(file=self.weights_location, arr=self.weights)

        else:
            # If not using weights, use an isotropic weight matrix
            print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.one_hot_encoding.shape[0]

        print("Neff =", str(self.Neff))
        print("Data Shape =", self.one_hot_encoding.shape)

    def create_all_singles(self):
        start_idx = self.focus_start_loc
        focus_seq_index = 0
        self.mutant_to_letter_pos_idx_focus_list = {}
        list_valid_mutations = []
        # find all possible valid mutations that can be run with this alignment
        alphabet_set = set(list(self.alphabet))
        for i, letter in enumerate(self.focus_seq):
            if letter in alphabet_set and letter != "-":
                for mut in self.alphabet:
                    pos = start_idx + i
                    if mut != letter:
                        mutant = letter + str(pos) + mut
                        self.mutant_to_letter_pos_idx_focus_list[mutant] = [letter, pos, focus_seq_index]
                        list_valid_mutations.append(mutant)
                focus_seq_index += 1
        self.all_single_mutations = list_valid_mutations

    def save_all_singles(self, output_filename):
        with open(output_filename, "w") as output:
            output.write('mutations')
            for mutation in self.all_single_mutations:
                output.write('\n')
                output.write(mutation)


def generate_mutated_sequences(msa_data, list_mutations):
    """
    Copied from VAE_model.compute_evol_indices.

    Generate mutated sequences using a MSAProcessing data object and list of mutations of the form "A42T" where position
    42 on the wild type is changed from A to T.
    Multiple mutations are separated by colons e.g. "A42T:C9A"

    Returns a tuple (list_valid_mutations, valid_mutated_sequences),
    e.g. (['wt', 'A3T'], {'wt': 'AGAKLI', 'A3T': 'AGTKLI'})
    """
    list_valid_mutations = ['wt']
    valid_mutated_sequences = {}
    valid_mutated_sequences['wt'] = msa_data.focus_seq_trimmed  # first sequence in the list is the wild_type

    # Remove (multiple) mutations that are invalid
    for mutation in list_mutations:
        individual_substitutions = mutation.split(':')
        mutated_sequence = list(msa_data.focus_seq_trimmed)[:]
        fully_valid_mutation = True
        for mut in individual_substitutions:
            wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            if pos not in msa_data.uniprot_focus_col_to_wt_aa_dict \
                    or msa_data.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa \
                    or mut not in msa_data.mutant_to_letter_pos_idx_focus_list:
                print("Not a valid mutant: " + mutation)
                fully_valid_mutation = False
                break
            else:
                wt_aa, pos, idx_focus = msa_data.mutant_to_letter_pos_idx_focus_list[mut]
                mutated_sequence[idx_focus] = mut_aa  # perform the corresponding AA substitution

        if fully_valid_mutation:
            list_valid_mutations.append(mutation)
            valid_mutated_sequences[mutation] = ''.join(mutated_sequence)

    return list_valid_mutations, valid_mutated_sequences


# Copied from VAE_model.compute_evol_indices
# One-hot encoding of sequences
def one_hot_3D(seq_keys, seq_name_to_sequence, msa_data):
    """
    Take in a list of sequence names and corresponding sequences, and generate a one-hot array according to an alphabet.
    """
    aa_dict = {letter: i for (i, letter) in enumerate(msa_data.alphabet)}

    one_hot_out = np.zeros((len(seq_keys), len(msa_data.focus_cols), len(msa_data.alphabet)))
    for i, mutation in enumerate(seq_keys):
        sequence = seq_name_to_sequence[mutation]
        for j, letter in enumerate(sequence):
            if letter in aa_dict:
                k = aa_dict[letter]
                one_hot_out[i, j, k] = 1.0
    one_hot_out = torch.tensor(one_hot_out)
    return one_hot_out


def gen_one_hot_to_sequence(one_hot_tensor, alphabet):
    """Reverse of one_hot_3D. Need the msa_data again. Returns a list of sequences."""
    for seq_tensor in one_hot_tensor:  # iterate through outer dimension
        seq = ""
        letters_idx = seq_tensor.argmax(-1)

        for idx in letters_idx.tolist():  # Could also do map(di.get, letters_idx)
            letter = alphabet[idx]
            seq += letter
        yield seq


def one_hot_to_sequence_list(one_hot_tensor, alphabet):
    return list(gen_one_hot_to_sequence(one_hot_tensor, alphabet))


# Could maybe use numba.jit for this? And/or could translate to torch and use GPU if it fits in memory?
def compute_weight(seq, list_seq, theta, debug=False):
    # seq shape: (L * alphabet_size,)
    number_non_empty_positions = np.sum(seq)  # = np.dot(seq,seq), assuming it is a flattened one-hot matrix
    if debug:
        print("number_non_empty_positions: " + str(number_non_empty_positions))
    if number_non_empty_positions > 0:
        denom = np.dot(list_seq, seq) / number_non_empty_positions  # number_non_empty_positions = np.dot(seq,seq)
        # if debug:
        # print(np.where(denom > 1 - theta))
        # print("raw denom 0,1,2,3,4,5,6", denom[[0,1,2,3,4,5,6]]*number_non_empty_positions)
        denom = np.sum(denom >= 1 - theta)  # Lood: Changed > to >=
        return 1 / denom
    else:
        return 0.0  # return 0 weight if sequence is fully empty


def compute_weight_global(i):
    seq = list_seq_global[i]
    # seq shape: (L * alphabet_size,)
    number_non_empty_positions = np.sum(seq)  # = np.dot(seq,seq), assuming it is a flattened one-hot matrix
    if number_non_empty_positions > 0:
        denom = np.dot(list_seq_global,
                       seq) / number_non_empty_positions  # number_non_empty_positions = np.dot(seq,seq)
        # if debug:
        # print(np.where(denom > 1 - theta))
        # print("raw denom 0,1,2,3,4,5,6", denom[[0,1,2,3,4,5,6]]*number_non_empty_positions)
        denom = np.sum(denom >= 1 - theta_global)  # Lood: Changed > to >=
        return 1 / denom
    else:
        return 0.0  # return 0 weight if sequence is fully empty


def init_worker(list_seq, theta):
    # Initialize the worker process
    # Note: Using global is not ideal, but not sure how else
    # It should be safe since processes have private global variables
    global list_seq_global
    global theta_global
    list_seq_global = list_seq
    theta_global = theta


# Could compare to evcouplings's num_cluster implementation https://github.com/debbiemarkslab/EVcouplings/blob/develop/evcouplings/align/alignment.py#L1172
#  And take best of both worlds
def compute_sequence_weights(list_seq, theta, num_cpus=1):
    _N, _seq_len, _alphabet_size = list_seq.shape  # = len(self.seq_name_to_sequence.keys()), len(self.focus_cols), len(self.alphabet)
    list_seq = list_seq.reshape((_N, _seq_len * _alphabet_size))
    print(f"Using {num_cpus} cpus for EVE weights computation")

    # # Could maybe use numba.jit for this? And/or could translate to torch and use GPU if it fits in memory?
    # def compute_weight(seq, debug=False):
    #     # seq shape: (L * alphabet_size,)
    #     number_non_empty_positions = np.sum(seq)  # = np.dot(seq,seq), assuming it is a flattened one-hot matrix
    #     if debug:
    #         print("number_non_empty_positions: " + str(number_non_empty_positions))
    #     if number_non_empty_positions > 0:
    #         denom = np.dot(list_seq, seq) / number_non_empty_positions  # number_non_empty_positions = np.dot(seq,seq)
    #         # if debug:
    #             # print(np.where(denom > 1 - theta))
    #             # print("raw denom 0,1,2,3,4,5,6", denom[[0,1,2,3,4,5,6]]*number_non_empty_positions)
    #         denom = np.sum(denom >= 1 - theta)  # Lood: Changed > to >=
    #         return 1 / denom
    #     else:
    #         return 0.0  # return 0 weight if sequence is fully empty

    if num_cpus > 1:
        # Compute weights in parallel
        with multiprocessing.Pool(processes=num_cpus, initializer=init_worker, initargs=(list_seq, theta)) as pool:
            # func = functools.partial(compute_weight, list_seq=list_seq, theta=theta)
            chunksize = max(min(8, int(_N / num_cpus / 4)), 1)
            print("chunksize: " + str(chunksize))
            # imap: Lazy version of map
            # Parallel progress bars are complicated, so just used a single one
            weights_map = tqdm(pool.imap(compute_weight_global, range(_N), chunksize=chunksize),
                               total=_N)
            weights = np.array(list(weights_map))
    else:
        weights_map = map(lambda seq: compute_weight(seq, list_seq=list_seq, theta=theta), list_seq)
        weights = np.array(list(tqdm(weights_map, total=_N)))

    return weights


def is_empty_sequence_one_hot(one_hot_matrix):
    # Could also just use the literal value -1 or NaN, or GAP as in EVCouplings to denote empty in the original array?
    return one_hot_matrix.reshape(one_hot_matrix.shape[0], -1).sum()


def is_empty_sequence_matrix(matrix, empty_value):
    assert len(matrix.shape) == 2, f"Matrix must be 2D; shape={matrix.shape}"
    assert isinstance(empty_value, (int, float)), f"empty_value must be a number; type={type(empty_value)}"
    # Check for each sequence if all positions are equal to empty_value
    empty_idx = np.all((matrix == empty_value), axis=1)
    return empty_idx


@numba.jit(nopython=True)  # , fastmath=True, parallel=True)
def calc_num_clusters_i(matrix, identity_threshold, invalid_value, i: int, L_non_gaps: float):
    N, L = matrix.shape
    L_non_gaps = 1.0 * L_non_gaps  # Show numba it's a float

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_clusters_i = 1  # Self
    # compare all pairs of sequences
    for j in range(N):
        if i == j:
            continue
        pair_matches = 0
        for k in range(L):
            if matrix[i, k] == matrix[j, k] and matrix[
                i, k] != invalid_value:  # Edit(Lood): Don't count gaps as similar?
                pair_matches += 1
        # Edit(Lood): Calculate identity as fraction of non-gapped positions (so asymmetric)
        if pair_matches / L_non_gaps >= identity_threshold:
            num_clusters_i += 1

    return num_clusters_i


@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_cluster_members_nogaps_parallel(matrix, identity_threshold, invalid_value):
    """
    From EVCouplings: https://github.com/debbiemarkslab/EVcouplings
    Calculate number of sequences in alignment
    within given identity_threshold of each other
    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    invalid_value : int
        Value in matrix that is considered invalid, e.g. gap or lowercase character.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # debug_val = 289
    # tmp_pairs = []
    # compare all pairs of sequences
    # Edit: Rewrote loop without any dependencies between inner and outer loops, so that it can be parallelized
    for i in prange(N):
        num_neighbors_i = 0
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                if matrix[i, k] == matrix[j, k] and matrix[
                    i, k] != invalid_value:  # Edit(Lood): Don't count gaps as matches
                    pair_matches += 1
            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            if pair_matches / L_non_gaps[i] >= identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i

    return num_neighbors


# Below are util functions copied from EVCouplings: https://github.com/debbiemarkslab/EVcouplings
# This code looks slow but it's because it's written as a numba kernel
# Fastmath should be safe here, as we can assume that there are no NaNs in the input etc.
@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_cluster_members_nogaps(matrix, identity_threshold, invalid_value):
    """
    From EVCouplings: https://github.com/debbiemarkslab/EVcouplings
    Calculate number of sequences in alignment
    within given identity_threshold of each other
    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # debug_val = 289
    # tmp_pairs = []
    # compare all pairs of sequences
    for i in range(N - 1):
        for j in range(i + 1, N):
            pair_matches = 0
            for k in range(L):
                if matrix[i, k] == matrix[j, k] and matrix[
                    i, k] != invalid_value:  # Edit(Lood): Don't count gaps as similar?
                    pair_matches += 1

            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so asymmetric)
            if pair_matches / L_non_gaps[i] >= identity_threshold:
                num_neighbors[i] += 1
                # if i == debug_val:
                #     tmp_pairs.append(j)
            if pair_matches / L_non_gaps[j] >= identity_threshold:
                num_neighbors[j] += 1
                # if j == debug_val:
                #     tmp_pairs.append(i)
            # elif i == debug_val or j == debug_val:
            #     print("Skipping pair:", "(", i, ",", j, ") ",
            #           pair_matches, "/", "L_non_gaps[i]:", L_non_gaps[i], "=", pair_matches / L_non_gaps[i],
            #           pair_matches, "/", "L_non_gaps[j]:", L_non_gaps[j], "=", pair_matches / L_non_gaps[j])

    return num_neighbors


@numba.jit(nopython=True, parallel=True)
def num_cluster_members_from_pair(matrix, identity_threshold, invalid_value, L_i):
    N = matrix.shape[0]
    pairs_matrix = np.zeros((N))
    for i in prange(N):
        pairs_matrix[i] = calc_num_clusters_i(matrix, identity_threshold=identity_threshold,
                                              invalid_value=invalid_value, i=i,
                                              L_non_gaps=L_i[i])
    return pairs_matrix


# Much faster using prange, but not sure how to tell numba we only got given a subset of cpus by slurm?
@numba.jit(nopython=True, parallel=True)
def func_all_i(matrix, identity_threshold, invalid_value, L_i):
    N = matrix.shape[0]
    num_clusters = np.zeros(N)
    for i in prange(N):
        num_clusters[i] = calc_num_clusters_i(matrix, identity_threshold=identity_threshold,
                                              invalid_value=invalid_value, i=i, L_non_gaps=L_i[i])
    return num_clusters


def calc_weights_evcouplings(matrix_mapped, identity_threshold, empty_value, num_cpus=1):
    """
        From EVCouplings: https://github.com/debbiemarkslab/EVcouplings
        Calculate weights for sequences in alignment by
        clustering all sequences with sequence identity
        greater or equal to the given threshold.
        Parameters
        ----------
        identity_threshold : float
            Sequence identity threshold
        """
    empty_idx = is_empty_sequence_matrix(matrix_mapped,
                                         empty_value=empty_value)  # e.g. sequences with just gaps or lowercase, no valid AAs

    # Original EVCouplings code structure, plus gap handling
    if num_cpus > 1:
        # This works on the datasets I've tested on, but probably a good idea to report it anyway
        print("Calculating weights using Numba parallel (experimental) since num_cpus > 1. "
              "If you want to disable multiprocessing set num_cpus=1.")  
        print("Default number of threads for Numba:", numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(num_cpus)
        print("Set number of threads to:", numba.get_num_threads())
        num_cluster_members = calc_num_cluster_members_nogaps_parallel(matrix_mapped[~empty_idx], identity_threshold,
                                                                       invalid_value=empty_value)
    else:
        num_cluster_members = calc_num_cluster_members_nogaps(matrix_mapped[~empty_idx], identity_threshold,
                                                              invalid_value=empty_value)

    # Empty sequences: weight 0
    N = matrix_mapped.shape[0]
    weights = np.zeros((N))
    weights[~empty_idx] = 1.0 / num_cluster_members
    return weights


def map_from_alphabet(alphabet, default=GAP):
    """
    Creates a mapping dictionary from a given alphabet.
    Parameters
    ----------
    alphabet : str
        Alphabet for remapping. Elements will
        be remapped according to alphabet starting
        from 0
    default : Elements in matrix that are not
        contained in alphabet will be treated as
        this character
    Raises
    ------
    ValueError
        For invalid default character
    """
    map_ = {
        c: i for i, c in enumerate(alphabet)
    }

    try:
        default = map_[default]
    except KeyError:
        raise ValueError(
            "Default {} is not in alphabet {}".format(default, alphabet)
        )

    return defaultdict(lambda: default, map_)


def map_matrix(matrix, map_):
    """
    Map elements in a numpy array using alphabet
    Parameters
    ----------
    matrix : np.array
        Matrix that should be remapped
    map_ : defaultdict
        Map that will be applied to matrix elements
    Returns
    -------
    np.array
        Remapped matrix
    """
    return np.vectorize(map_.__getitem__)(matrix)


########################
# Failed JIT parallel ideas:
# N, L = matrix_mapped[~empty_idx].shape
# L_i = L - np.sum(matrix_mapped[~empty_idx] == empty_value, axis=1)

# Idea 1: Calculate the full pair matrix (i,j) = number of matches between sequence i and sequence j
# neighbour_matrix = calc_num_pairs(matrix_mapped[~empty_idx], identity_threshold, invalid_value=empty_value)
# num_cluster_members = (pairs_matrix / L_i[:, None] >= identity_threshold).sum(axis=1)  # TODO check this axis

# Idea 2: Calculate the pair matrix but applying thresholding inside the loop
# num_cluster_members = neighbour_matrix.sum(axis=1)
# num_cluster_members = calc_num_cluster_members_nogaps_all_vs_all(
#     matrix_mapped[~empty_idx], identity_threshold, invalid_value=empty_value  # matrix_mapped[~empty_idx]
# )

# Idea 3)
# Multiprocessing with numba jit:
# def init_worker_ev(matrix, empty_value, identity_threshold):
#     global matrix_mapped_global
#     matrix_mapped_global = matrix
#     L = matrix.shape[1]
#     global empty_value_global
#     empty_value_global = empty_value
#     global identity_threshold_global
#     identity_threshold_global = identity_threshold
#     global L_i_global
#     L_i_global = L - np.sum(matrix == empty_value, axis=1)
#     print("Initialising worker")
#     global global_calc_num_clusters_i
#     global_calc_num_clusters_i = _global_calc_cluster_factory()
#     try:
#         _ = global_calc_num_clusters_i(0)
#     except Exception as e:
#         print("Worker initialisation failed:", e)
#         raise e
#     print("Function compiled")
#
#
# def _worker_func(i):
#     return global_calc_num_clusters_i(i)
#
#
# def _global_calc_cluster_factory():
#     @numba.jit(nopython=True)
#     def func(i):
#         return calc_num_clusters_i(matrix_mapped_global, identity_threshold_global, empty_value_global, i, L_non_gaps=L_i_global[i])
#     # out = calc_num_clusters_i(matrix=matrix_mapped_global, identity_threshold=identity_threshold_global, invalid_value=empty_value_global, i=i, L_non_gaps=L_i_global[i])
#     # print("Calculated, sending back ", out)
#     # return out
#     return func

# Inside calling function calc_weights_evcouplings_parallel
# if num_cpus > 1:
# Compute weights in parallel
# num_cpus = 1  # tmp
# print("Num CPUs for EVCouplings code:", num_cpus)
# with multiprocessing.Pool(processes=num_cpus, initializer=init_worker_ev, initargs=(matrix_mapped[~empty_idx], empty_value, identity_threshold)) as pool:
#     # func = functools.partial(compute_weight, list_seq=list_seq, theta=theta)
#     chunksize = max(min(32, int(N / num_cpus / 4)), 1)
#     print("chunksize: " + str(chunksize))
#     # imap: Lazy version of map
#     # Parallel progress bars are complicated
#     cluster_map = tqdm(pool.imap(_worker_func, range(N), chunksize=chunksize), total=N)

@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_pairs(matrix, identity_threshold, invalid_value):
    """
    From EVCouplings: https://github.com/debbiemarkslab/EVcouplings
    Calculate number of sequences in alignment
    within given identity_threshold of each other
    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    # L = 1.0 * L  # need to tell numba that L is a float

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    # L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    neighbour_matrix = np.eye(N)  # dtype=np.bool
    # debug_val = 289
    # tmp_pairs = []
    # Crucial: We assume none of the sequences are empty
    # Construct a loop that counts a neighbour if the pairwise identity is above the threshold
    pairs_j = np.zeros(N, dtype=np.int32)
    for i in range(N):
        # Calculate the non-gapped length of sequence i
        # L_i = np.sum(matrix[i] != invalid_value)  # Can either use L_i or L_j to calculate the neighbor matrix, the output will simply be transposed
        pairs_j[:] = 0
        for j in range(N):
            num_pairs = 0
            for k in range(L):
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                    num_pairs += 1
            pairs_j[j] = num_pairs  # Could also just add this as an array at the end of j loop
        neighbour_matrix[i] = pairs_j  # Could also calc identity threshold here

    return neighbour_matrix
