# Basically train_VAE.py but just calculating the weights
import argparse
import os
import time

import numpy as np
import pandas as pd

from utils import data_utils
from utils.data_utils import GAP, ALPHABET_PROTEIN_GAP
from utils.weights import calc_weights_evcouplings, map_matrix, map_from_alphabet


def create_argparser():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored', required=True)
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name', required=True)
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file', required=True)
    parser.add_argument('--MSA_weights_location', type=str,
                        help='Location where weights for each sequence in the MSA will be stored', required=True)
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument("--num_cpus", type=int, help="Number of CPUs to use", default=1)
    # Note: It would be nicer to have an overwrite flag, but I don't want to change the MSAProcessing code too much
    parser.add_argument("--skip_existing", help="Will quit gracefully if weights file already exists", action="store_true", default=False)
    parser.add_argument("--calc_method", choices=["evcouplings", "eve", "both"], help="Method to use for calculating weights", default="evcouplings")
    parser.add_argument("--calc_speedup", help="Print debug information", action="store_true", default=False)
    return parser


def main(args):
    print("Arguments:", args)

    assert os.path.isfile(args.MSA_list), f"MSA file list {args.MSA_list} doesn't seem to exist"
    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: " + str(protein_name))
    print("MSA file: " + str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
        print(f"Using custom theta value {theta} instead of loading from mapping file.")
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except KeyError as e:
            # Overriding previous errors is bad, but we're being nice to the user
            raise KeyError("Couldn't load theta from mapping file. "
                           "NOT using default value of theta=0.2; please specify theta manually. Specific line:",
                           mapping_file[args.protein_index],
                           "Previous error:", e)

    print("Theta MSA re-weighting: " + str(theta))

    if not os.path.isdir(args.MSA_weights_location):
        # exist_ok=True: Otherwise we'll get some race conditions between concurrent jobs
        os.makedirs(args.MSA_weights_location, exist_ok=True)
        print(f"{args.MSA_weights_location} is not a directory. "
              f"Being nice and creating it for you, but this might be a mistake.")
        # raise NotADirectoryError(f"{args.MSA_weights_location} is not a directory."
        #                          f"Could create it automatically, but at the moment raising an error.")

    weights_file = args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    # First check that the weights file doesn't exist
    if os.path.isfile(weights_file):
        if args.skip_existing:
            print("Weights file already exists, skipping")
            exit(0)
        else:
            raise FileExistsError(f"File {weights_file} already exists. "
                                  f"Please delete it if you want to re-calculate it. "
                                  f"If you want to skip existing files, use --skip_existing.")

    msa_data = data_utils.MSA_processing(
        MSA_location=msa_location,
        theta=theta,
        use_weights=False, #True,
        weights_location=weights_file,
        num_cpus=args.num_cpus,
    )

    msa_data.use_weights = True  # Manually set this so that we can check smaller runtimes first
    num_cpus = args.num_cpus
    
    # Below is for testing, will be removed from calc_weights.py
    if args.calc_method == "evcouplings" or args.calc_method == "both":
        alphabet_mapper = map_from_alphabet(ALPHABET_PROTEIN_GAP, default=GAP)
        arrays = []
        for seq in msa_data.seq_name_to_sequence.values():
            arrays.append(np.array(list(seq)))
        sequences = np.vstack(arrays)
        sequences_mapped = map_matrix(sequences, alphabet_mapper)

        print("Compiling JIT function")
        start = time.perf_counter()
        # TODO could also check the speedup here?
        # Compile jit function
        _ = calc_weights_evcouplings(sequences_mapped[:10], identity_threshold=1 - theta,
                                     empty_value=0, num_cpus=num_cpus)  # GAP = 0
        print("JIT function compiled/run in {} seconds".format(time.perf_counter() - start))
        if args.calc_speedup:
            # del sequences
            print("Checking runtime for JIT function with different args")
            start = time.perf_counter()
            _ = calc_weights_evcouplings(sequences_mapped[:11], identity_threshold=1 - theta,
                                         empty_value=0, num_cpus=num_cpus)  # GAP = 0
            print("JIT function with length 11 run in {} seconds".format(time.perf_counter() - start))

            print("100 seqs:")
            start = time.perf_counter()
            _ = calc_weights_evcouplings(sequences_mapped[:100], identity_threshold=1 - theta,
                                         empty_value=0, num_cpus=num_cpus)  # GAP = 0
            print("JIT function with length 100 run in {} seconds".format(time.perf_counter() - start))
            ###########################################################
            print("1000 seqs:")
            start = time.perf_counter()
            _ = calc_weights_evcouplings(sequences_mapped[:1000], identity_threshold=1 - theta,
                                         empty_value=0, num_cpus=num_cpus)  # GAP = 0
            print("JIT function with length 1000 run in {} seconds".format(time.perf_counter() - start))
            #######################################################################################################
            print("10k seqs:")
            start = time.perf_counter()
            _ = calc_weights_evcouplings(sequences_mapped[:10000], identity_threshold=1 - theta,
                                         empty_value=0, num_cpus=num_cpus)  # GAP = 0
            print("JIT function with length 10k run in {} seconds".format(time.perf_counter() - start))
            ##################################
            print("100k seqs:")
            start = time.perf_counter()
            _ = calc_weights_evcouplings(sequences_mapped[:100000], identity_threshold=1 - theta,
                                         empty_value=0, num_cpus=num_cpus)  # GAP = 0
            print("JIT function with length 100k run in {} seconds".format(time.perf_counter() - start))
            runtime_parallel = time.perf_counter() - start
            # Init/compile EVCouplings weights calc for 1 CPU (different path than parallel)
            print("Checking runtime for 1 cpu")
            _ = calc_weights_evcouplings(sequences_mapped[:100000], identity_threshold=1 - theta,
                                         empty_value=0, num_cpus=1)  # GAP = 0
            start_1cpu = time.perf_counter()
            _ = calc_weights_evcouplings(sequences_mapped[:100000], identity_threshold=1 - theta,
                                         empty_value=0, num_cpus=1)  # GAP = 0
            runtime_1cpu = time.perf_counter() - start_1cpu
            print("1 CPU JIT function with length 100k run in {} seconds".format(runtime_1cpu))
            print(f"Speedup: {runtime_1cpu / runtime_parallel:.2f}x for {num_cpus} CPUs")

        # Calculate and save weights
        ev = msa_data.calc_weights(num_cpus=num_cpus, method="evcouplings")

    if args.calc_method == "eve" or args.calc_method == "both":
        eve = msa_data.calc_weights(num_cpus=num_cpus, method="eve")
    if args.calc_method == "both":
        # tmp check diffs
        for old, new, seq, idx in zip(eve[eve != ev], ev[eve != ev], sequences[eve != ev],
                                      np.where(eve != ev)[0]):
            print(f"Sequence {idx} {''.join(seq)}: EVE {1 / old} -> ev {1 / new}")

        assert np.array_equal(eve, ev), f"EVCouplings and EVE weights are not equal. EVcouplings weights: {ev}, EVE weights: {eve}"
        # del sequences_mapped
        print("EVCouplings and EVE weights are equal")


if __name__ == '__main__':
    start = time.perf_counter()
    parser = create_argparser()
    args = parser.parse_args()
    main(args)
    end = time.perf_counter()
    print(f"calc_weights.py took {end-start:.2f} seconds in total.")
