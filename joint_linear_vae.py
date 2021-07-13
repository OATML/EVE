# Copied from semi-supervised.ipynb notebook
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# Hacky relative import
# %cd ../EVE
# !pip install ../EVE/
# !pip list
from EVE.VAE_model import VAE_model
from utils import data_utils


# If we're outside of proj-protein_latent/visualisation
# %cd ./proj-protein_latent/visualisation
# %cd -

# TODO save new model checkpoints
def main(model_checkpoint,
         msa_location,
         natural_labels_path,
         model_parameters_location,
         training_logs_location,
         plot_save_dir=None,
         # Cross-validation options
         lm_training_frequency=2,
         lm_elbo_together=True,  # TODO On steps optimizing the linear model, also optimise the VAE loss
         num_training_steps=200,
         prev_num_steps=400000,  # Previous number of steps the model was trained for
         ):
    ####################

    protein_name = "ADRB2"

    MSA_LOCATION = msa_location
    NATURAL_LABELS_PATH = natural_labels_path
    PLOT_SAVE_DIR = plot_save_dir

    os.makedirs(training_logs_location, exist_ok=True)

    assert os.path.isfile(model_parameters_location), model_parameters_location

    if PLOT_SAVE_DIR:
        assert os.path.exists(PLOT_SAVE_DIR), PLOT_SAVE_DIR

    assert os.path.exists(MSA_LOCATION), MSA_LOCATION

    ##################

    start = time.time()
    msa_data = data_utils.MSA_processing(
                MSA_location=MSA_LOCATION,
                theta=0.2,
                use_weights=False,
    #             weights_location=args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'  # Weights are saved here during training
    )
    print(f"Time taken: {(time.time()-start)//60}m:{(time.time()-start)%60:.3f}s", )

    #####################################

    msa_df = pd.DataFrame.from_dict(msa_data.seq_name_to_sequence, orient='index', columns=['sequence'])
    msa_df['Uniref'] = msa_df.index
    msa_df.reset_index(drop=True)
    msa_df['Uniref'] = msa_df['Uniref'].apply(lambda s: s.replace(">", ""))  # Strip the > character

    #################################

    # MSA labels
    assert os.path.isfile(NATURAL_LABELS_PATH)
    natural_labels_df = pd.read_csv(NATURAL_LABELS_PATH)
    print(len(natural_labels_df), "rows")
    # display(natural_labels_df.head())

    msa_merged_df = pd.merge(left=msa_df, right=natural_labels_df, how='left', on='Uniref')
    print(len(msa_merged_df), 'rows')

    ############################

    model_params = json.load(open(model_parameters_location))

    MODEL_CHECKPOINT = model_checkpoint
    assert os.path.isfile(MODEL_CHECKPOINT), f"{MODEL_CHECKPOINT} is not a file."
    assert os.stat(MODEL_CHECKPOINT).st_size != 0, "File is empty, this will cause errors."

    # Load model checkpoint
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=torch.device("cpu"))

    def load_model(model_name=protein_name):
        # First, init model with random weights
        vae_model = VAE_model(
            model_name=model_name,
            data=msa_data,
            encoder_parameters=model_params["encoder_parameters"],
            decoder_parameters=model_params["decoder_parameters"],
            random_seed=42,
        )
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        # vae_model.eval() # Turn off dropout etc.
        return vae_model



    ###########################

    # Add auxiliary linear regression
    class VAEAux(torch.nn.Module):
        def __init__(self, vae_model, linear_out_features):
            super().__init__()
            self.vae_model = vae_model
            linear_in = model_params['encoder_parameters']['z_dim']
            self.linear_out_features = linear_out_features
            self.lm = torch.nn.Linear(linear_in, linear_out_features)

        def forward(self, x):
            mu, log_var = self.vae_model.encoder(x)
            z = self.vae_model.sample_latent(mu, log_var)
            return mu, log_var, z, self.lm(z)

        def encode_and_predict(self, x):
            mu, log_var = self.vae_model.encoder(x)
            z = self.vae_model.sample_latent(mu, log_var)
            #         lm_pred_interval = torch.clamp(lm_pred, -2, 0)
            return self.lm(z)

        def predict_lm(self, z):
            return self.lm(z)


    # Add auxiliary loss

    # Train

    ########################

    training_parameters = checkpoint['training_parameters']

    #################################

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    targets = torch.as_tensor(msa_merged_df['GNAS'].values).to(device).view(-1, 1)  # Many NaNs, only 128 labels

    print("Joint training taking turns between unsupervised and labelled samples")


    # def evaluate(x, y):
    #     vae_aux.eval()
    #
    #     mu, log_var, z, lm_pred = vae_aux.forward(x)
    #     recon_x_log = vae_aux.vae_model.decoder(z)
    #     neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = vae_aux.vae_model.loss_function(recon_x_log, x, mu,
    #                                                                                                log_var,
    #                                                                                                training_parameters[
    #                                                                                                    'kl_latent_scale'],
    #                                                                                                training_parameters[
    #                                                                                                    'kl_global_params_scale'],
    #                                                                                                training_parameters[
    #                                                                                                    'annealing_warm_up'],
    #                                                                                                training_step,
    #                                                                                                Neff_training)
    #
    #     mse = torch.nn.MSELoss()(lm_pred, y)  # Can also do sigmoid loss for soft classification from -2 to 0?
    #     print("test mse: ", mse.item())
    #     #     print("test r2", r2_score(y, lm_pred.detach().numpy()))
    #
    #     return lm_pred.detach().numpy(), mse

    def evaluate_sampled(x, y, num_samples=20):
        assert y.isnan().sum() == 0
        # TODO can pack multiple x_test together in a batch?
        y_pred_all = torch.cat([vae_aux.forward(x)[3].detach() for _ in
                                range(num_samples)])  # vae_aux.forward returns (mu, log_var, z, lm_pred)
        y_test_all = torch.cat([y for _ in range(num_samples)])
        mse_all = torch.nn.MSELoss()(y_test_all, y_pred_all)
        print("Total test mse:", mse_all)
        # print("Total test R^2:", r2_score(y_test_all, y_pred_all))
        # TODO can also get mean, std of mse/R^2 across samples if stacked
        return y_pred_all, y_test_all, mse_all

    for fold_idx, (train_index, val_index) in enumerate(kf.split(msa_data.one_hot_encoding)):
        print(f"CV fold {fold_idx}")
        torch.manual_seed(42)
        vae_model = load_model(model_name=protein_name + "lood_linear_regression_joint")
        vae_aux = VAEAux(vae_model, linear_out_features=1)
        # All parameters on for joint training
        vae_aux.to(device)
        vae_aux.train()

        print("tmp weights init:", vae_aux.lm.weight)

        # Optimize over all parameters (VAE + prediction model)
        optimizer = torch.optim.Adam(vae_aux.parameters(), lr=training_parameters['learning_rate'],
                                     weight_decay=training_parameters['l2_regularization'])
        vae_aux.train()

        x_train = torch.as_tensor(msa_data.one_hot_encoding[train_index], dtype=torch.float, device=device)
        weights_train = msa_data.weights[train_index]
        y_train = torch.as_tensor(targets[train_index], dtype=torch.float, device=device)

        train_label_mask = ~targets[train_index].isnan().any(dim=-1)
        x_train_labeled = torch.as_tensor(msa_data.one_hot_encoding[train_index][train_label_mask], dtype=torch.float,
                                          device=device)
        y_train_labeled = torch.as_tensor(targets[train_index][train_label_mask], dtype=torch.float, device=device)

        # TODO also use the test set to get validation ELBO curve
        x_test = torch.as_tensor(msa_data.one_hot_encoding[val_index], dtype=torch.float, device=device)
        weights_test = msa_data.weights[val_index]
        y_test = torch.as_tensor(targets[val_index], dtype=torch.float, device=device)

        test_label_mask = ~targets[val_index].isnan().any(dim=-1)
        x_test_labeled = torch.as_tensor(msa_data.one_hot_encoding[val_index][test_label_mask], dtype=torch.float,
                                         device=device)
        y_test_labeled = torch.as_tensor(targets[val_index][test_label_mask], dtype=torch.float, device=device)

        batch_order = np.arange(x_train.shape[0])
        seq_sample_probs = weights_train / np.sum(weights_train)
        Neff_training = np.sum(weights_train)

        training_metrics = {"mse": [], "neg_ELBO": [], "BCE": []}  # "r2": [],
        validation_metrics = {"mse": []}

        for training_step in tqdm.tqdm(range(1, num_training_steps+1), desc="Training linear reg model"):
            optimizer.zero_grad()

            # Linear model + joint training
            if training_step % lm_training_frequency == 0:
                x, y = x_train_labeled, y_train_labeled

                mu, log_var, z, lm_pred = vae_aux.forward(x)
                mse = torch.nn.MSELoss()(lm_pred, y)  # Can also do sigmoid loss for soft classification from -2 to 0?
                loss = 10*mse

                # Random thought: Can you optimize encoder and decoder separately? e.g. KL div one step, recon_x next step? In this case we might want to just optimize encoder + linear model.
                if lm_elbo_together:
                    recon_x_log = vae_aux.vae_model.decoder(z)
                    neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = vae_aux.vae_model.loss_function(recon_x_log, x, mu, log_var, training_parameters['kl_latent_scale'], training_parameters['kl_global_params_scale'], training_parameters['annealing_warm_up'], prev_num_steps + training_step, Neff_training)
                    loss = neg_ELBO + 10 * mse  # Can weight these appropriately: Since lr * 100 worked before, can we just do loss*100?

                    training_metrics["neg_ELBO"].append(neg_ELBO.item())
                    training_metrics["BCE"].append(BCE.item())

                training_metrics["mse"].append(mse.item())
                print(training_step, "Training mse:", mse.item())
            #             print(training_step, "Training R^2:", r2_score(y, lm_pred.detach().numpy()))
            else:
                # Sample a batch according to sequence weight (note: this will also affect the weights of the regression?)
                batch_sample_index = np.random.choice(batch_order, training_parameters['batch_size'],
                                                      p=seq_sample_probs).tolist()

                x = torch.as_tensor(x_train[batch_sample_index], dtype=torch.float, device=device)
                # y = torch.as_tensor(y_train[batch_sample_index], dtype=torch.float, device=device)

                # Unsupervised training
                mu, log_var, z, lm_pred = vae_aux.forward(x)
                recon_x_log = vae_aux.vae_model.decoder(z)
                neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = vae_aux.vae_model.loss_function(recon_x_log, x,
                                                                                                           mu, log_var,
                                                                                                           training_parameters[
                                                                                                               'kl_latent_scale'],
                                                                                                           training_parameters[
                                                                                                               'kl_global_params_scale'],
                                                                                                           training_parameters[
                                                                                                               'annealing_warm_up'],
                                                                                                           prev_num_steps + training_step,
                                                                                                           Neff_training)

                loss = neg_ELBO

                training_metrics["neg_ELBO"].append(neg_ELBO.item())
                training_metrics["BCE"].append(BCE.item())

            loss.backward()
            optimizer.step()

            if training_step % 10 == 0:
                #             y_pred, mse = evaluate(x_test, y_test)
                _, _, mse = evaluate_sampled(x_test_labeled, y_test_labeled, num_samples=5)
                validation_metrics['mse'].append(mse.item())

        print("weights after training:\n", vae_aux.lm.weight, "bias:", vae_aux.lm.bias)

        if PLOT_SAVE_DIR:
            for metric in training_metrics:
                plt.plot(training_metrics[metric])
                plt.title("Training " + metric)
                plt.savefig(os.path.join(PLOT_SAVE_DIR, f"training_{metric}_fold_{fold_idx}"))
                plt.clf()  # Clear figure, never knew this before
            for metric in validation_metrics:
                plt.plot(validation_metrics[metric])
                plt.title("Validation " + metric)
                plt.savefig(os.path.join(PLOT_SAVE_DIR, f"test_{metric}_fold_{fold_idx}"))
                plt.clf()  # Clear figure, never knew this before

        # Aggregate predictions
        num_samples = 20
        y_pred_all, y_test_all, mse = evaluate_sampled(x_test_labeled, y_test_labeled, num_samples=num_samples)

        # Also write results out to CSV
        csv_path = os.path.join(training_logs_location, f"test_fold_{fold_idx}.csv")
        assert len(y_pred_all.cpu().numpy()) > 0, y_pred_all.cpu().numpy()
        print("tmp", y_pred_all.cpu().numpy()[:10])
        print("tmp", y_test_all.cpu().numpy()[:10])

        df = pd.DataFrame.from_dict({'pred': y_pred_all.cpu().numpy(), 'test': y_test_all.cpu().numpy()}, orient='columns')
        df.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    # parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name will be the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--training_logs_location', type=str, help='Where results (txt, csv) should be written')
    parser.add_argument('--labels_path', type=str, help='Labels for linear regression in this case')
    # Cross-validation/training options
    parser.add_argument('--num_training_steps', type=int, help="Number of steps of fine-tuning")
    parser.add_argument('--lm_elbo_together', type=int)  # 0 or 1
    parser.add_argument('--lm_training_frequency', type=int)
    args = parser.parse_args()

    print("tmp: args=", args)

    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: " + str(protein_name))
    print("MSA file: " + str(msa_location))

    fine_tuning_kwargs = {}
    if args.num_training_steps is not None:
        fine_tuning_kwargs['num_training_steps'] = args.num_training_steps
    if args.lm_elbo_together is not None:
        fine_tuning_kwargs['lm_elbo_together'] = args.lm_elbo_together
    if args.lm_training_frequency is not None:
        fine_tuning_kwargs['lm_training_frequency'] = args.lm_training_frequency

    main(model_checkpoint=args.VAE_checkpoint_location,
         msa_location=msa_location,
         natural_labels_path=args.labels_path,
         model_parameters_location=args.model_parameters_location,
         training_logs_location=args.training_logs_location,
         plot_save_dir=args.training_logs_location,
         **fine_tuning_kwargs)
