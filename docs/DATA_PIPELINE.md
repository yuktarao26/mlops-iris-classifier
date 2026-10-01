# Data Pipeline Documentation

## Pipeline Stages

| Stage | Purpose | Input | Output |
|---|---|---|---|
| Collect | Obtain raw Iris data | sklearn Iris dataset | `iris_raw.csv` |
| Preprocess | Clean data | `iris_raw.csv` | `iris_preprocessed.csv` |
| Feature Engineering | Create useful features | `iris_preprocessed.csv` | `iris_features.csv` |
| Validate | Check schema, nulls and ranges | `iris_features.csv` | Validation result |

## Pipeline Flow

Collect → Preprocess → Feature Engineering → Validate

## Validation Rules

- Expected columns must be present.
- No unexpected null values.
- Species values must be valid.
- Numeric features must be within the expected ranges.