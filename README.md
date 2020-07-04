# Quantum Warehouse
> Reinforcement Learning architecture to automate long term planning of warehouse inventory for enterprise deployment.

## MongoDB Database Setup
> Navigate to data/config.py

```
USERNAME = "YOUR_USERNAME"
PASSWORD = "YOUR_PASSWORD"
DATABASE_NAME = "warehouse"
```

## Installation Instructions (macOS & Linux)
> Python 3.7.4 should be installed in the system. 
> You should be outside the root directory `Quantum-Warehouse`

```
python3 -m venv env 
source env/bin/activate
pip install -r Quantum-Warehouse/requirements.txt
```

```
cd Quantum-Warehouse
python main.py
```