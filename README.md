Datasets and code for `RDNet: Robust Detection of Malicious Encrypted Traffic against Obfuscation through Spatial-Temporal Feature Fusion Network`.

## Datasets

1、The MTA dataset (malware-traffic-analysis) can be obtained at [malware-traffic-analysis.net](https://www.malware-traffic-analysis.net/index.html).

2、The Stratosphere dataset can be obtained at [Malware Capture Facility Project]( https://www.stratosphereips.org/datasets-malware).

3、The CIC-IOT-2023 dataset can be obtained at [Canadian Institute for Cybersecurity]( https://www.unb.ca/cic/datasets/iotdataset-2023.html).

## Directory Structure
1、datset: We provide a sample of the processed MTA dataset. The dataset uses the improved PLD algorithm and has been modeled as an AFG graph structure. The dataset is stored as a bin file and a pkl file.

2、code：We have integrated the data reading and classification modules into a single Python file.

## How to clone
`git clone https://github.com/CN-RDNet/RDNet.git`

`git lfs pull`


## Environment
python 3.9

requirement：`requirement.txt`

## How to run
`cd code`

`python RDNet.py`
