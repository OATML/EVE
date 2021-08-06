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

from EVE.VAE_model import VAE_model
from utils import data_utils


def save_vae_aux(vae_aux, checkpoint_path, encoder_parameters, decoder_parameters, training_parameters):
        # Create intermediate dirs above this
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        # Could also just save the vae and linear model separately
        torch.save({
            'model_state_dict': vae_aux.state_dict(),
            'encoder_parameters': encoder_parameters,
            'decoder_parameters': decoder_parameters,
            'training_parameters': training_parameters,
        }, checkpoint_path)


def main(model_checkpoint,
         msa_location,
         natural_labels_path,
         model_params,
         training_logs_location,
         plot_save_dir=None,
         # Cross-validation options
         lm_training_frequency=2,
         lm_elbo_together=True,  # TODO On steps optimizing the linear model, also optimise the VAE loss
         num_training_steps=200,
         prev_num_steps=400000,  # Previous number of steps the model was trained for
         training_mode='mixed',
         load_checkpoint=True,
         lm_loss_weight=10,
         ):
    ####################

    protein_name = "ADRB2"

    MSA_LOCATION = msa_location
    NATURAL_LABELS_PATH = natural_labels_path
    PLOT_SAVE_DIR = plot_save_dir
    training_parameters = model_params['training_parameters']

    # Change training parameters in-place so that we can save the correct parameters in the checkpoint
    training_parameters['learning_rate'] = training_parameters['learning_rate'] * 1
    if not load_checkpoint:
        prev_num_steps = 0

    use_mean_embeddings = False
    # print("Using mean embeddings")
    shrink_init_variance = False
    loss_fn = "mse"

    os.makedirs(training_logs_location, exist_ok=True)

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

    if load_checkpoint:
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
        if load_checkpoint:
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

    ########################

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    cols = ['GNAS']
    # cols = ['GNAS', 'GNAL', 'GNAI1', 'GNAI3', 'GNAO1', 'GNAZ', 'GNAQ', 'GNA14', 'GNA15', 'GNA12', 'GNA13']

    targets = torch.as_tensor(msa_merged_df[cols].values).to(device).view(-1, len(cols))  # Many NaNs, only 128 labels

    def evaluate_sampled(x, y, num_samples=20):
        assert y.isnan().sum() == 0
        y_pred_all = torch.cat([vae_aux.forward(x)[3].detach() for _ in
                                range(num_samples)])  # vae_aux.forward returns (mu, log_var, z, lm_pred)
        y_test_all = torch.cat([y for _ in range(num_samples)])
        mse_all = torch.nn.MSELoss()(y_test_all, y_pred_all)
        print("Total test mse:", mse_all)
        # print("Total test R^2:", r2_score(y_test_all, y_pred_all))
        # TODO can also get mean, std of mse/R^2 across samples if stacked

        bce_linear_all = torch.nn.BCELoss()(torch.sigmoid(y_pred_all), y_test_all / 2 + 1)
        print("Total test linear bce:", bce_linear_all)

        return y_pred_all, y_test_all, mse_all

    for fold_idx, (train_index, val_index) in enumerate(kf.split(msa_data.one_hot_encoding)):
        print(f"CV fold {fold_idx}")
        torch.manual_seed(42)
        vae_model = load_model(model_name=protein_name + "lood_linear_regression_joint")
        vae_aux = VAEAux(vae_model, linear_out_features=len(cols))
        # All parameters on for joint training
        vae_aux.to(device)
        vae_aux.train()
        if training_mode == "frozen":
            vae_aux.eval()
            vae_aux.lm.train()

        print("tmp weights init:", vae_aux.lm.weight, vae_aux.lm.bias)

        # Init assumes Var(output) = 1. Here output variance is bigger, so let's scale down by a factor
        if shrink_init_variance:
            with torch.no_grad():
                vae_aux.lm.weight.data = vae_aux.lm.weight.clone() / 10.

        # Optimize over all parameters (VAE + prediction model)
        optimizer = torch.optim.Adam(vae_aux.parameters(), lr=training_parameters['learning_rate'],  # Increase learning rate; increase linear model loss weight
                                     weight_decay=training_parameters['l2_regularization'])
        if training_mode == "frozen":
            print("Frozen encoder/decoder: only optimizing over lm parameters")
            optimizer = torch.optim.Adam(vae_aux.lm.parameters(), lr=training_parameters['learning_rate'],
                                         weight_decay=training_parameters['l2_regularization'])

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

        training_metrics = {"mse": [], "neg_ELBO": [], "BCE": [], "bce_linear": []}  # "r2": [],
        validation_metrics = {"mse": [], 'bce_linear': []}

        # Init the bias for better convergence?
        print("y_train_labeled shape:", y_train_labeled.size())
        assert y_train_labeled.size()[1] == len(cols), y_train_labeled.size()
        vae_aux.lm.bias.data = torch.mean(y_train_labeled, dim=0)
        mean = torch.mean(y_train_labeled, dim=0, keepdim=True)
        print("mean(y_train): ", mean.detach().cpu())
        print("mean shape before: ", mean.size())
        baseline_pred = mean.expand(y_test_labeled.size())  # Broadcast mean up to N predictions, match shape for MSE loss
        print("baseline shape:", baseline_pred.size())
        print("Baseline mse (by predicting only mean): per-component:", cols, torch.nn.MSELoss(reduction='none')(baseline_pred, y_test_labeled).mean(dim=0), "reduced:", torch.nn.MSELoss()(baseline_pred, y_test_labeled))
        # print("Baseline BCE/log-loss (by predicting only mean):", torch.nn.BCELoss()(baseline_pred / 2 + 1, y_test_labeled / 2 + 1))

        # TODO aggregate mean/std of CV scores

        # TODO should probably trim out the logging and keep outside functions
        def mixed_batch_loss():
            lm_l2_regularization = 0
            # Train together in same batch
            prop_batch_labeled = 0.75
            num_labeled = int(training_parameters['batch_size'] * prop_batch_labeled)

            # Sample labeled
            sample_index_labeled = np.random.randint(0, x_train_labeled.shape[0], size=num_labeled).tolist()
            batch_labeled = x_train_labeled[sample_index_labeled]
            batch_labeled_y = y_train_labeled[sample_index_labeled]

            # Sample unlabeled
            sample_index_unlabeled = np.random.choice(batch_order, training_parameters['batch_size'] - num_labeled,
                                                      p=seq_sample_probs).tolist()
            batch_unlabeled = x_train[sample_index_unlabeled]

            batch = torch.cat((batch_labeled, batch_unlabeled), dim=0)
            assert batch.size()[0] == training_parameters['batch_size']

            mu, log_var = vae_aux.vae_model.encoder(batch)
            z = vae_aux.vae_model.sample_latent(mu, log_var)
            if use_mean_embeddings:
                lm_pred = vae_aux.lm(mu)
            else:
                lm_pred = vae_aux.lm(z)
            recon_x_log = vae_aux.vae_model.decoder(z)
            neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = vae_aux.vae_model.loss_function(recon_x_log,
                                                                                                       batch, mu,
                                                                                                       log_var,
                                                                                                       training_parameters[
                                                                                                           'kl_latent_scale'],
                                                                                                       training_parameters[
                                                                                                           'kl_global_params_scale'],
                                                                                                       training_parameters[
                                                                                                           'annealing_warm_up'],
                                                                                                       prev_num_steps + training_step,
                                                                                                       Neff_training)

            lm_l2_norm = torch.norm(vae_aux.lm.weight, p=2)

            if loss_fn == "mse":
                mse = torch.nn.MSELoss()(lm_pred[:num_labeled], batch_labeled_y)
                loss = neg_ELBO + lm_loss_weight * mse + lm_l2_regularization * lm_l2_norm
                training_metrics["mse"].append(mse.item())
                print(training_step, "Training mse:", mse.item())
            elif loss_fn == "sigmoid":
                lm_pred = torch.sigmoid(lm_pred[:num_labeled])
                y_sig = batch_labeled_y / 2 + 1
                bce = torch.nn.BCELoss()(lm_pred, y_sig)
                loss = neg_ELBO + lm_loss_weight * bce + lm_l2_regularization * lm_l2_norm
                training_metrics["bce_linear"].append(bce.item())
                print(training_step, "Training bce:", bce.item())
            else:
                raise ValueError(f"loss_fn must be one of [mse,sigmoid]. {loss_fn} given")

            training_metrics["neg_ELBO"].append(neg_ELBO.item())
            training_metrics["BCE"].append(BCE.item())
            return loss

        def alternating_loss(training_step):
            # Linear model + joint training
            if training_step % lm_training_frequency == 0:
                x, y = x_train_labeled, y_train_labeled

                mu, log_var = vae_aux.vae_model.encoder(x_train_labeled)
                z = vae_aux.vae_model.sample_latent(mu, log_var)
                if use_mean_embeddings:
                    lm_pred = vae_aux.lm(mu)
                else:
                    lm_pred = vae_aux.lm(z)
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
                # Sample a batch according to sequence weight
                batch_sample_index = np.random.choice(batch_order, training_parameters['batch_size'],
                                                      p=seq_sample_probs).tolist()

                x = x_train[batch_sample_index]
                # y = y_train[batch_sample_index]

                # Unsupervised training
                mu, log_var = vae_aux.vae_model.encoder(x)
                z = vae_aux.vae_model.sample_latent(mu, log_var)
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

            return loss

        def frozen_vae_loss():
            x, y = x_train_labeled, y_train_labeled
            with torch.no_grad():
                mu, log_var = vae_aux.vae_model.encoder(x_train_labeled)
                z = vae_aux.vae_model.sample_latent(mu, log_var)
            if use_mean_embeddings:
                lm_pred = vae_aux.lm(mu)
            else:
                lm_pred = vae_aux.lm(z)
            if loss_fn == "mse":
                mse = torch.nn.MSELoss()(lm_pred, y)  # Can also do sigmoid loss for soft classification from -2 to 0?
                loss = 10 * mse
                training_metrics["mse"].append(mse.item())
                print(training_step, "Training mse:", mse.item())
            elif loss_fn == "sigmoid":
                lm_pred = torch.sigmoid(lm_pred)
                y_sig = y / 2 + 1
                bce = torch.nn.BCELoss()(lm_pred, y_sig)
                loss = 10 * bce
                training_metrics["bce_linear"].append(bce.item())
                print(training_step, "Training bce:", bce.item())
            else:
                raise ValueError(f"loss_fn must be one of [mse,sigmoid]. {loss_fn} given")

            return loss

        for training_step in tqdm.tqdm(range(1, num_training_steps+1), desc="Training linear reg model"):
            optimizer.zero_grad()

            if training_mode == "mixed":
                loss = mixed_batch_loss()
            elif training_mode == "alternating":
                loss = alternating_loss(training_step)
            elif training_mode == "frozen":
                loss = frozen_vae_loss()
            else:
                raise KeyError(f"Training mode must be 'mixed', 'alternating' or 'frozen'. {training_mode} given")

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
                plt.title(f"Training {metric}: Fold {fold_idx}")
                plt.savefig(os.path.join(PLOT_SAVE_DIR, f"training_{metric}_fold_{fold_idx}"))
                plt.clf()  # Clear figure, never knew this before
            for metric in validation_metrics:
                plt.plot(validation_metrics[metric])
                plt.title(f"Validation {metric}: Fold {fold_idx}")
                plt.savefig(os.path.join(PLOT_SAVE_DIR, f"test_{metric}_fold_{fold_idx}"))
                plt.clf()  # Clear figure, never knew this before

        # Aggregate predictions
        num_samples = 20
        print("Final test mse:")
        y_pred_all, y_test_all, mse = evaluate_sampled(x_test_labeled, y_test_labeled, num_samples=num_samples)

        # Also write results out to CSV
        csv_path = os.path.join(training_logs_location, f"test_fold_{fold_idx}.csv")
        assert len(y_pred_all.cpu().numpy()) > 0, y_pred_all.cpu().numpy()

        # using .squeeze() to remove the extra dimensions of size 1 e.g. [N, 1] -> [N], because pandas requires 1D arrays
        if len(cols) == 1:
            df = pd.DataFrame.from_dict({'pred': y_pred_all.cpu().numpy().squeeze(), 'test': y_test_all.cpu().numpy().squeeze()}, orient='columns')
            df.to_csv(csv_path)
        # For multiple columns, can save an e.g. GNAS_pred, GNAS_test column pair for each column
        checkpoint_out = os.path.join(training_logs_location, f"fold_{fold_idx}")
        save_vae_aux(vae_aux,
                     checkpoint_path=checkpoint_out,
                     encoder_parameters=model_params["encoder_parameters"],
                     decoder_parameters=model_params["decoder_parameters"],
                     training_parameters=training_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    # parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location of pretrained VAE model chekpoint')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name will be the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--training_logs_location', type=str, help='Where results (txt, csv) should be written')
    parser.add_argument('--labels_path', type=str, help='Labels for linear regression in this case')
    # Cross-validation/training options
    # Note: It would be nice to take in arbitrary arguments here and pass them straight through to the main function instead of marshalling them along.
    parser.add_argument('--num_training_steps', type=int, help="Number of steps of fine-tuning")
    parser.add_argument('--z_dim', type=int, help='Specify a different latent dim than in the params file')
    parser.add_argument('--lm_elbo_together', type=int)  # 0 or 1
    parser.add_argument('--lm_training_frequency', type=int)
    parser.add_argument('--training_mode')  # 'mixed' or 'alternating'
    parser.add_argument('--lm_loss_weight', type=float)
    args = parser.parse_args()

    print("tmp: args=", args)

    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: " + str(protein_name))
    print("MSA file: " + str(msa_location))

    fine_tuning_kwargs = {}
    # TODO use args.dict instead
    if args.num_training_steps is not None:
        fine_tuning_kwargs['num_training_steps'] = args.num_training_steps
    if args.lm_elbo_together is not None:
        fine_tuning_kwargs['lm_elbo_together'] = args.lm_elbo_together
    if args.lm_training_frequency is not None:
        fine_tuning_kwargs['lm_training_frequency'] = args.lm_training_frequency
    if args.training_mode is not None:
        fine_tuning_kwargs['training_mode'] = args.training_mode
    if args.lm_loss_weight is not None:
        fine_tuning_kwargs['lm_loss_weight'] = args.lm_loss_weight

    model_params = json.load(open(args.model_parameters_location))

    # Overwrite params if necessary
    if args.z_dim:
        model_params["encoder_parameters"]["z_dim"] = args.z_dim
        model_params["decoder_parameters"]["z_dim"] = args.z_dim

    main(model_checkpoint=args.VAE_checkpoint_location,
         msa_location=msa_location,
         natural_labels_path=args.labels_path,
         model_params=model_params,
         training_logs_location=args.training_logs_location,
         plot_save_dir=args.training_logs_location,
         load_checkpoint=False,  # TODO tmp, should rather read this in from command line
         **fine_tuning_kwargs)
