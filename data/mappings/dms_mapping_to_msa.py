import pandas as pd

dms_mapping = pd.read_csv("data/mappings/DMS_mapping_20220227.csv")
df_weights = dms_mapping[['UniProt_ID', 'MSA_filename', 'theta']].drop_duplicates().replace({'UniProt_ID': 'protein_name', 'MSA_filename': 'msa_location'})
df_weights.to_csv("mapping_msa_tkmer_20220227.csv")
