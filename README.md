# Biometric Verification Performance Analysis (FAR / FRR / EER)

Evaluates a biometric verification system by comparing genuine and impostor match scores, then computing the False Accept Rate, False Reject Rate and Equal Error Rate.

Course: **BSM361 – Introduction to Biometric Systems**, Bartın University

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c)

## Result

| Metric | Value |
|---|---|
| Equal Error Rate (EER) | **4.67%** |
| EER threshold | 0.8637 |

## Method

1. **Data**: `Features.npz` holds a feature array of shape `(10 sessions, 110 people, 6 features)`; the first 100 people are used.
2. **Normalization**: min-max scaling to the 0–1 range.
3. **Scores**: Euclidean distance `d` between feature vectors, converted to a similarity score `1 / (1 + d)`.
   - **Genuine**: same person, different sessions
   - **Impostor**: different people, same session
4. **Evaluation**: 500 thresholds between 0 and 1; at each one, FAR and FRR are computed and the EER is taken where they cross.

## Plots

The script produces three figures:
- Genuine vs impostor score distributions
- FAR and FRR against the threshold, with the EER point marked
- FAR vs FRR (DET-style) curve

## Run

```bash
pip install numpy scipy matplotlib
python biyometrik_odev1.py
```

The repository also includes an OpenCV image basics notebook (`2. Image Basics with OpenCV/`).
