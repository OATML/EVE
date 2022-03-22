import os, sys
import argparse
import pandas as pd
import json

from EVE import VAE_model
from utils import data_utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored', required=True)
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name', required=True)
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file', required=True)
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored', required=True)
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored', required=True)
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name will be the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--training_logs_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--z_dim', type=int, help='Specify a different latent dim than in the params file')
    parser.add_argument('--force_load_weights', action='store_true', help="Force loading of weights from MSA_weights_location (useful if you want to make sure you're using precalculated weights). Will fail if weight file doesn't exist.", default=False)

    args = parser.parse_args()

    print("Arguments:", args)

    assert os.path.isfile(args.MSA_list), f"MSA file list {args.MSA_list} doesn't seem to exist"
    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: "+str(protein_name))
    print("MSA file: "+str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            print("Couldn't load theta from mapping file. Using default value of 0.2")
            theta = 0.2
    print("Theta MSA re-weighting: "+str(theta))


    weights_file = args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    if args.force_load_weights:
        print("Flag force_load_weights enabled - Forcing that we use weights from file:", weights_file)
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f"Weights file {weights_file} doesn't exist. "
                                    f"To recompute weights, remove the flag --force_load_weights.")

    data = data_utils.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            weights_location=weights_file,
    )

    model_name = protein_name + "_" + args.model_name_suffix
    print("Model name: "+str(model_name))

    assert os.path.isfile(args.model_parameters_location), args.model_parameters_location
    model_params = json.load(open(args.model_parameters_location))

    # Overwrite params if necessary
    if args.z_dim:
        model_params["encoder_parameters"]["z_dim"] = args.z_dim
        model_params["decoder_parameters"]["z_dim"] = args.z_dim

    model = VAE_model.VAE_model(
                    model_name=model_name,
                    data=data,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    random_seed=42
    )
    model = model.to(model.device)

    model_params["training_parameters"]['training_logs_location'] = args.training_logs_location
    model_params["training_parameters"]['model_checkpoint_location'] = args.VAE_checkpoint_location
    
    # model_params["training_parameters"]["num_training_steps"] = 1000 # todo temp debugging

    print("Starting to train model: " + model_name)
    model.train_model(data=data, training_parameters=model_params["training_parameters"])

    print("Saving model: " + model_name)
    model.save(model_checkpoint=model_params["training_parameters"]['model_checkpoint_location']+os.sep+model_name+"_final", 
                encoder_parameters=model_params["encoder_parameters"], 
                decoder_parameters=model_params["decoder_parameters"], 
                training_parameters=model_params["training_parameters"]
    )
