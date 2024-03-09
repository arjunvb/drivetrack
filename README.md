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

First, obtain access to the [Waymo Open Dataset ](https://waymo.com/open/) if you don't have access already.

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

## Attribution
DriveTrack was made using the Waymo Open Dataset, provide by Waymo LLC under license terms available at [waymo.com/open](https://waymo.com/open/).

## Citation
If you use this code or our data for your research, please cite:

**DriveTrack: A Benchmark for Long-Range Point Tracking in Real-World Videos**\
Arjun Balasingam, Joseph Chandler, Chenning Li, Zhoutong Zhang, Hari Balakrishnan.\
_To appear at CVPR 2024_

```
@inproceedings{balasingam2024drivetrack,
 author = {Arjun Balasingam and Joseph Chandler and Chenning Li and Zhoutong Zhang and Hari Balakrishnan},
 title = {DriveTrack: A Benchmark for Long-Range Point Tracking in Real-World Videos},
 booktitle = {CVPR},
 year = {2024}
}
```
