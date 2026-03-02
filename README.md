# ML Device Fault Detection – IEEE ML Challenge

## Team

* Aadi Joshi — Model Training
* Neer Joshi — Prediction Pipeline & Submission

---

## Problem

Predict whether a device is **faulty (1)** or **normal (0)** using 47 numerical sensor features.

---

## Approach

* Train/validation split on TRAIN.csv
* XGBoost binary classification model
* Validation accuracy: **98.39%**
* Saved trained model artifact
* Automated prediction pipeline for TEST.csv

---

## Repository Structure

```
ML-Device-Fault-Detection
│
├── data/           # datasets (not tracked)
├── src/
│   ├── train.py    # model training (Aadi)
│   └── predict.py  # prediction & submission (Neer)
├── outputs/        # model & submission
├── notebooks/      # experiments
├── requirements.txt
└── README.md
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python src/train.py
```

Outputs:

```
outputs/model.pkl
```

---

## Generate Submission

```bash
python src/predict.py
```

Outputs:

```
outputs/FINAL.csv
```

---

## Submission Format

```
ID,Class
1,1
2,0
...
```

---

## Notes

* Data files are excluded via `.gitignore`
* Pipeline is fully reproducible
* Model trained on TRAIN.csv and evaluated on validation split
