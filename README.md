# GO-VAE-Model-Explorer

Instructions on how to download and run the web application for exploring model results.

1. Clone the repository

```
git clone https://github.com/daria-dc/GO-VAE-Model-Explorer.git
```

2. Download the data from dropbox under the following link

https://www.dropbox.com/s/yq6eatp10l7a4em/data.tar.gz?dl=0


3. Store the data folder in the repo folder. You should now have the following structure:

```
.
├── app-GTEx.py
├── assets
│   ├── bootstrap.slate.css
│   └── custom.css
├── data
│   ├── default_UMAP_results.csv
│   ├── latent_space_embedding.npy
│   ├── onto_trimmed_annot.csv
│   ├── onto_trimmed_graph.json
│   ├── onto_trimmed_wang_sem_sim.npy
│   ├── sample_annot.csv
│   └── Wilcox_results.csv
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
