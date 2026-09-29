# DVC Workflow

## DVC Remote Storage
The project uses a local DVC remote storage:

~/dvc-remote-storage

## Dataset Versioning Workflow

1. Add dataset using DVC:
   dvc add data/raw/iris_v1.csv

2. Track the DVC metadata with Git:
   git add data/raw/iris_v1.csv.dvc
   git commit -m "data: add dataset"

3. Push dataset to DVC remote:
   dvc push

## Comparing Dataset Versions

DVC can compare dataset versions using:

dvc diff <commit_hash>

The `.dvc` file stores the hash of the dataset version.

## Reproducing Historical Dataset Versions

Checkout the required `.dvc` file:

git checkout <commit_hash> -- data/raw/iris_v1.csv.dvc

Then restore the dataset:

dvc checkout data/raw/iris_v1.csv.dvc

This allows previous dataset versions to be reproduced.

## Version History

- Version 1: 150 rows
- Version 2: 170 rows

DVC stores dataset contents separately while Git tracks the dataset version metadata.