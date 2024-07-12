# Targeting AAV vectors to the CNS via de novo engineered capsid-receptor interactions
Code and documentation supporting "Targeting AAV vectors to the CNS via de novo engineered capsid-receptor interactions", including data, SVAE-based variant generation method, and figure-generation code.

## Contents

# Installation
Code is provided as a collection of [Jupyter Notebooks](https://jupyter.org/) and requires `python3.8` or `python3.9`.

Install dependencies:

```
# Dependencies are pinned for python3.8/3.9

pip3 install -r requirements.txt
```


### SVAE model and training data

1. **Training data processing** - [`AAV_capsid_receptor/notebooks/pulldown_assay_data_processing.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/notebooks/pulldown_assay_data_processing.ipynb)
   
   - Starting from read counts, compute reads per million (RPM) and $\log_2$ enrichment for LY6A-Fc and LY6C1-Fc; export a CSV of `mean_RPM`, `cv_RPM` (coefficient of variation), and `log2enr` values for each of LY6A-Fc and LY6C1-Fc.

2. **SVAE model and variant generation** (for LY6C1-Fc) - [`AAV_capsid_receptor/notebooks/SVAE_variant_generation.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/notebooks/SVAE_variant_generation.ipynb)
    
    1. Starting from the CSV exported by `pulldown_assay_data_processing.ipynb`, format LY6C1-Fc data into TensorFlow-compatible training batches. 
    
    2. Initialize and train an SVAE model.
    
    3. Cluster and sample the trained SVAE model's latent space to generate novel variants.

### Paper figures

**Note:** all figure-generation notebooks assume figure data is contained in `AAV_capsid_receptor/data` (see [Data](#data) for more details).

1. Figure 1 and 1S (supplemental) panels - [`AAV_capsid_receptor/figures/fig1.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/figures/fig1.ipynb)
2. Figure 2 and 2S (supplemental) panels - [`AAV_capsid_receptor/figures/fig2.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/figures/fig2.ipynb)
3. Figure 3 panels - [`AAV_capsid_receptor/figures/fig3.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/figures/fig3.ipynb)
4. Figure 4 and 4S (supplemental) panels - [`AAV_capsid_receptor/figures/fig4.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/figures/fig4.ipynb)


# Data

All relevant data is stored on Zenodo at [DOI 10.5281/zenodo.8222089](https://doi.org/10.5281/zenodo.8222089). Once downloaded, data files should be put into `AAV_capsid_receptor/data` - by default, figure-generation notebooks will search for data there.

