# GO-VAE-Model-Explorer

Instructions on how to download and run the web application for exploring model results.

1. Clone the repository

```
git clone https://github.com/daria-dc/GO-VAE-Model-Explorer.git
```

2. Download the data from dropbox under the following link

https://www.dropbox.com/s/l5kjyo5goqj72wi/data.tar.gz?dl=0


3. Store the data folder in the repo folder. You should now have the following structure:

```
.
├── app-GTEx.py
├── assets
│   ├── bootstrap.slate.css
│   └── custom.css
├── data
│   ├── GO_BP_at_least_depth3_min_30_max_1000_trimmed_annot.csv
│   ├── GO_BP_recount3_genes_depth3_min_30_max_1000_trimmed_graph.json
│   ├── go_ids_depth3_trimmed_wang_sem_sim.npy
│   ├── GTEx_quant_norm_trimmed_ensemble100_avg_GO_activations.npy
│   ├── GTEx_samples_latent_space_embedding_RVAE-484.npy
│   ├── recount3_depth3_min_30_max_1000_trimmed_GO_genes.csv
│   ├── recount_brain_GTEx_TCGA_annot.csv
│   ├── RVAE-484_UMAP_results.csv
│   └── trimmed_GTEx_ensemble100_brain_vs_rest_GO_node_activations.csv
└── requirements.txt
```


 
3. Create a python virtual environment or conda environment and install necessary packages with 

```
pip install -r requirements.txt
```

4. Inside of this environment, run the following command

```
python3 app-GTEx.py
```
