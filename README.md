# Hand Reco

This repository contains the code for the hand recognition project. This project is in three parts:
1. Hand detection using Mediapipe
|--- a. Hand detection & capture the landmarks of the hand
|--- b. Store the landmarks in a csv file
2. Hand recognition using Machine Learning
3. Live hand recognition using OpenCV and Mediapipe

## Installation

1. Clone the repository

```bash
git clone https://github.com/rohitprasad-code/hand-reco.git && cd hand-reco
```

2. Install the conda environment

```bash
conda create -n hand-reco python=3.10 -y
conda activate hand-reco
```

3. Install the requirements

```bash
pip install -r requirements.txt
```

4. Run the code for data collection

```bash
python data_collection.py
```

5. Run the juptyer notebook for training the model

```bash
jupyter notebook
```

6. Run the code for live hand recognition

```bash
python live_hand_reco.py
```
