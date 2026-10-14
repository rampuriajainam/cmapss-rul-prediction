# Remaining Useful Life Prediction on NASA C-MAPSS (FD001)

Predicting how many cycles a turbofan engine has left before failure, from its 21 sensor readings and 3 operating settings, on the FD001 subset of NASA's C-MAPSS dataset. A baseline comparison of a Random Forest and an LSTM, plus an analysis of where the models fail.

## Setup
- Target: remaining useful life (RUL) = last cycle of the engine − current cycle, **not capped**.
- Split by engine, not by row (80/20), so no engine appears in both training and validation.
- For the LSTM, inputs are standardised with statistics from the training engines only.

## Results (validation engines)

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest, single cycle | 26.0 | 35.6 |
| Random Forest, flattened 20-cycle window | 21.9 | 30.3 |
| LSTM baseline (20-cycle window, 2 layers) | 21.2 | 29.9 |
| **LSTM tuned** (1 layer, hidden 64, lr 0.002, batch 64) | **19.7** | **27.5** |

Tuning changed one hyperparameter at a time against the baseline: hidden size (64/128/256), layers and dropout, learning rate (0.0005–0.004) and batch size (32/64/128). The full tables are in `notebooks/02_lstm.ipynb`.

## What the analysis showed

**1. The models cannot predict long remaining life.** For validation samples with true RUL above 160 (mean 193), the LSTM predicted 145 on average and never above about 150. Early in an engine's life the sensors barely change, so there is little signal to tell RUL 200 from RUL 150, and the error concentrates there. On samples with RUL ≤ 80, the same model reaches MAE 10.7 and RMSE 16.9.

**2. Longer windows look better partly for the wrong reason.** Increasing the window from 20 to 180 cycles cut the overall MAE from about 21 to 4.5, but long windows also drop every early-life sample with high RUL from validation. Comparing only on RUL ≤ 80 still shows a real gain (MAE 11.5 → 3.3), with the caveat that sample counts shrink (1620 → 625).

## Next steps
- Cap the RUL target (piecewise-linear, around 125 cycles), the standard choice on C-MAPSS, since early-life RUL is not predictable from the sensors.
- Evaluate on the official test set (`test_FD001` + `RUL_FD001`) for numbers comparable with published work.
- Try attention or 1D-CNN sequence models.

## Running it
Download C-MAPSS from NASA's Prognostics Data Repository and put `train_FD001.txt` next to the notebooks. The data is not included here.

```bash
pip install -r requirements.txt
```
