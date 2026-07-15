# DSME
DSME implementation

Generalizing Sensor-Based Human Activity Recognition With Domain-Specific Ensemble Learning

DOI: 10.1109/JSEN.2025.3549881
2025.05.06 

1. torchmetrics - install through pip

2. tqdm - install through conda


3. pytorch2.4.0- install through pytorch official page with conda
   1. pytorch 1.13 or lower versions are ok too

## Core Function of DSME

train_test_moe_with_torchmetrics() 

Code explanation: __Coming Soon__


## To preprocess data

1. Create folder "har_data/"
2. Visit whichever dataset repo and download *.zip or *.rar or xxx and move to "har_data/" and unzip or unrar or xxx
3. Directory tree should be
    1. har_data/
       1. e.g. DSADS/
       2. e.g. OpportunityUCIDataset/dataset/
       3. e.g. PAMAP2/Protocol/
    2. DSME
4. get to xxx/DSME/ and "bash ./scripts/preprocess.sh"
## To run
1. get to xxx/DSME/
   * python main.py with args
   * python cmd_generotor.py to get python commands

