# WHEN: Wavelet-DTW Hybrid Attention Network

Implementation and review of the WHEN architecture proposed in Wang et al. (2023),
submitted as an assigment at IITM.

---

## ⚠️ Implementation Status

**The WHEN architecture in this repository is a partial implementation and does not
fully reproduce the results of the original paper.**

The main known limitation concerns the **WaveAtt module**: the bandwidth parameter α
(Eq. 3 in the paper) frequently collapses to values close to zero during training.
As a consequence, the daughter wavelets computed in Eq. 4 take on very large values,
causing numerical instability. Batch normalization layers were added to the TD-NN-1
and TD-NN-2 modules to partially mitigate this issue, but the root cause in WaveAtt
was not resolved. Refer to my report for further details.

Additionally, the original paper uses 18 wavelet families, while this implementation
uses only 8, due to memory constraints.

Despite these limitations, the model is functional and produces reasonable results
on some datasets. A full discussion of the experimental outcomes is provided in the
accompanying report.

---

## Repository Structure

```
WHEN-classification/
├── README.md
├── requirements.txt
├── WHEN_Classification.ipynb      # main notebook: training + benchmarks
├── report.pdf                     # full project report
└── data/
    ├── ElectricDevices_TRAIN.ts
    ├── ElectricDevices_TEST.ts
    ├── SharePriceIncrease_TRAIN.ts
    ├── SharePriceIncrease_TEST.ts
    ├── WordSynonyms_TRAIN.ts
    └── WordSynonyms_TEST.ts
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Open the notebook
Open `WHEN_Classification.ipynb` in Jupyter or Google Colab and run the cells
in order. The first cell installs the required packages automatically.

### 3. Select the dataset
In **Cell 2**, set the `NAME` variable to choose which dataset to use:

```python
NAME = "ElectricDevices"      # 8926 train, 7711 test, 96 timesteps, 7 classes
# NAME = "SharePriceIncrease" #  965 train,  965 test, 60 timesteps, 2 classes
# NAME = "WordSynonyms"       #  267 train,  638 test, 270 timesteps, 25 classes
```

The corresponding `.ts` files must be present in the `data/` folder.

### 4. Training vs. inference
- If a `.pth` file matching `{NAME}_best_model_class.pth` is found in the folder,
  the notebook loads the pre-trained weights and skips training.
- Otherwise, training starts from scratch with early stopping (patience = 7).

> ⚠️ **Training is slow.** On a standard GPU, ElectricDevices takes approximately
> 1 hour per epoch. WordSynonyms takes over 10 minutes per epoch.
> Running the benchmark cell on a pre-trained model is much faster.

### 5. Benchmarks
**Cell 6** runs all baselines automatically:
- **ROCKET** — convolutional kernel transform (gold standard)
- **K-NN** with DTW distance and RBF kernel
- **SVC** with DTW distance and RBF kernel
- **Random Forest** with rolling statistical features
- **WHEN** — loaded from the saved `.pth`

**Cell 7** prints the results table (LaTeX-ready) and saves the confusion matrices.

---

## Datasets

All datasets are from the
[UEA Multivariate Time Series Classification Archive](http://www.timeseriesclassification.com/).
They are included in the `data/` folder for convenience.

---

## Reference

Wang, Jingyuan et al. (2023). *WHEN: A Wavelet-DTW Hybrid Attention Network
for Heterogeneous Time Series Analysis*. In Proceedings of the 29th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining (KDD '23), Long Beach, CA.
DOI: [10.1145/3580305.3599549](https://doi.org/10.1145/3580305.3599549)
