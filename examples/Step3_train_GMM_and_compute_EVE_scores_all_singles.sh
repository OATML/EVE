export input_evol_indices_location='./results/evol_indices'
export input_evol_indices_filename_suffix='_20000_samples'
export protein_list='./data/mappings/example_mapping.csv'
export output_eve_scores_location='./results/EVE_scores'
export output_eve_scores_filename_suffix='Jan1_PTEN_example'

export GMM_parameter_location='./results/GMM_parameters/Default_GMM_parameters'
export GMM_parameter_filename_suffix='default'
export protein_GMM_weight=0.3
export plot_location='./results'
export labels_file_location='./data/labels/PTEN_ClinVar_labels.csv'
export default_uncertainty_threshold_file_location='./utils/default_uncertainty_threshold.json'

python train_GMM_and_compute_EVE_scores.py \
    --input_evol_indices_location ${input_evol_indices_location} \
    --input_evol_indices_filename_suffix ${input_evol_indices_filename_suffix} \
    --protein_list ${protein_list} \
    --output_eve_scores_location ${output_eve_scores_location} \
    --output_eve_scores_filename_suffix ${output_eve_scores_filename_suffix} \
    --load_GMM_models \
    --GMM_parameter_location ${GMM_parameter_location} \
    --GMM_parameter_filename_suffix ${GMM_parameter_filename_suffix} \
    --compute_EVE_scores \
    --protein_GMM_weight ${protein_GMM_weight} \
    --plot_histograms \
    --plot_scores_vs_labels \
    --plot_location ${plot_location} \
    --labels_file_location ${labels_file_location} \
    --default_uncertainty_threshold_file_location ${default_uncertainty_threshold_file_location} \
    --verbose
    
        