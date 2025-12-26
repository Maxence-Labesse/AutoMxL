# Améliorations proposées — Configuration (param_config.py)

## Priorité 1 : Constantes en UPPER_SNAKE_CASE
- Quoi : Renommer les constantes selon la convention
  - `batch_size` → `BATCH_SIZE`
  - `n_epoch` → `N_EPOCH`
  - `learning_rate` → `LEARNING_RATE`
  - `crit` → `CRITERION`
  - `optim` → `OPTIMIZER`
- Pourquoi : Respect des conventions de nommage (§3.2)
- Fichiers impactés : `param_config.py`, `Preprocessing/Deep_Encoder.py`, `Preprocessing/Categorical.py`

## Priorité 2 : Renommer les abréviations
- Quoi : `crit` → `CRITERION`, `optim` → `OPTIMIZER`
- Pourquoi : Nommage explicite (conventions §3.1)
- Fichiers impactés : `param_config.py`, modules qui importent ces constantes
