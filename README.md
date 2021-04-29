# Evolutionary model of Variant Effects (EVE)

This repository contains the code to create EVE models as per our paper "Large-scale clinical interpretation of genetic variants using evolutionary data and deep learning" (https://www.biorxiv.org/content/10.1101/2020.12.21.423785v1).

## Overview
EVE is a set of protein-specific models providing for any single amino acid mutation of interest a score reflecting the propensity of the resulting protein to be pathogenic. For each protein family, a Bayesian VAE learns a distribution over amino acid sequences from evolutionary data. It enables the computation of an evolutionary index for each mutant, which approximates the log-likelihood ratio of the mutant vs the wild type. A global-local mixture of Gaussian Mixture Models separates variants into benign and pathogenic clusters based on that index. The EVE scores reflect probabilistic assignments to the pathogenic cluster.

## Usage
The end to end process to compute EVE scores consists of three consecutive steps:
1. Train the Bayesian VAE on a re-weighted multiple sequence alignment (MSA) for the protein of interest => train_VAE.py
2. Compute the evolutionary indices for all single amino acid mutations => compute_evol_indices.py
3. Train a GMM to cluster variants on the basis of the evol indices then output scores and uncertainties on the class assignments => train_GMM_and_compute_EVE_scores.py
We also provide all EVE scores for all single amino acid mutations for thousands of proteins at the following address: http://evemodel.org/.

## Example scripts
The "examples" folder contains sample bash scripts to obtain EVE scores for the PTEN protein.
The corresponding MSA and ClinVar labels are provided in the data folder.

## Data requirements
The only data required to train EVE models and obtain scores from scratch are the multiple sequence alignments for the corresponding proteins. 
The third script (train_GMM_and_compute_EVE_scores.py) provides functionalities to compare EVE scores with reference labels (e.g., ClinVar) to be provided by the user.

## Software requirements
The entire codebase is written in python. Package requirements are as follows:
  - python=3.7
  - pytorch=1.7
  - cudatoolkit=11.0
  - scikit-learn=0.24.1
  - numpy=1.20.1
  - pandas=1.2.4
  - scipy=1.6.2
  - tqdm
  - matplotlib
  - seaborn

The corresponding environment may be created via conda and the provided protein_env.yml file as follows:
```
  conda env create -f protein_env.yml
  conda activate protein_env
```

## License
This project is available under the MIT license.

## Reference
If you use this code, please cite the following paper:
```
Large-scale clinical interpretation of genetic variants using evolutionary data and deep learning
Jonathan Frazer, Pascal Notin, Mafalda Dias, Aidan Gomez, Kelly Brock, Yarin Gal, Debora S. Marks
bioRxiv 2020.12.21.423785
doi: https://doi.org/10.1101/2020.12.21.423785
```