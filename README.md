# ML-Driven 5G Handover Optimization

This project investigates machine-learning-based optimization of 5G handovers using ns-3 and 5G-LENA.

## Current Features

- Two-gNB 5G NR simulation
- One moving UE
- A3 RSRP handover
- X2 handover between gNBs
- UE mobility tracking
- RSRP and RSRQ measurements
- UE speed logging
- Serving and neighbor cell measurements
- Handover event tracing
- CSV dataset generation

## Current Dataset Features

The simulator currently generates:

- time
- UE position
- UE speed
- serving cell
- serving RSRP
- neighbor RSRP
- serving RSRQ
- neighbor RSRQ
- RSRP difference
- handover event
- target cell

## Project Goal

The goal is to train machine-learning models that can predict whether a UE should:

1. Stay connected to its current serving cell
2. Perform a handover
3. Select the appropriate target cell

The ML-based approach will later be compared with the traditional A3 RSRP handover algorithm.

## Technologies

- ns-3
- 5G-LENA
- C++
- Python
- Pandas
- Scikit-learn
- Machine Learning

## Project Structure

```text
5g-handover-ml/
├── simulation/
│   └── ml-handover.cc
├── data/
│   ├── raw/
│   │   └── handover-data.csv
│   └── processed/
├── ml/
├── models/
├── results/
├── README.md
└── .gitignore