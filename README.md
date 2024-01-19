# Drivetrack

## Installation

```sh
# Initialize the git submodules
git submodule init --update

# Install dependencies
conda env create -f environment.yml
conda activate drivetrack

# Build NLSPN DeformConv
cd nlspn/src/model/deformconv
sh make.sh
cd ../../../../

# Build CompletionFormer DeformConv
cd completionformer/src/model/deformconv
sh make.sh
cd ../../../../
```

In order to fix some bugs, you also need to upgrade the following packages after creating the conda environment:

```sh
pip install --upgrade numpy dask[distributed] pyarrow 
```

You will get warnings about dependencies not matching with waymo-open-dataset. It is safe to ignore these warnings.

## Getting Started

An example of generating DriveTrack (requires access to the Waymo Open Dataset in GCS):

```sh
python generate_drivetrack.py \
  --output-dir "output-path" \
  --use-gcsfs \
  --split "training" \
  --version "1.0.0" \
  --gpus 0,1,2
```

Alternatively, if you have the Waymo dataset downloaded locally, you can specify a local path instead of using the GCS bucket:

```sh
python generate_drivetrack.py \
  --output-dir "output-path" \
  --local-dataset-path "dataset-path" \
  --split "training" \
  --version "1.0.0" \
  --gpus 0,1,2
```

A full list of arguments can be found in `generate_drivetrack.py`.
