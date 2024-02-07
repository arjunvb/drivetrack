# Drivetrack

## Installation

```sh
# Initialize the git submodules
git submodule init --update

# Install dependencies
conda env create -f environment.yml
conda activate drivetrack
pip install -r requirements.txt --no-dependencies

# Build NLSPN DeformConv
cd nlspn/src/model/deformconv
sh make.sh
cd ../../../../

# Build CompletionFormer DeformConv
cd completionformer/src/model/deformconv
sh make.sh
cd ../../../../
```

Note that the depdencies that ship with the Waymo SDK need to be upgraded, which is why we install using pip and the `--no-dependencies` flag.

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
