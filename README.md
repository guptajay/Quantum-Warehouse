# Quantum Warehouse
> Reinforcement Learning architecture to automate long term planning of warehouse inventory for enterprise deployment.

## Installation Instructions

```
conda -V  
conda update conda
conda create -n quantumwarehouse python=3.7.4 anaconda
conda activate quantumwarehouse
python -m pip install keras keras-rl matplotlib gym tensorflow==1.13.1
```

```
cd "environment package/warehouse" // Navigate to Warehouse Environment
python setup.py install
```

```
cd .. // Navigate to Root 
cd .. // Navigate to Root
python main.py
```

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `ModuleNotFoundError: No module named 'warehouse'`

***

# API usage

Csv files can be found at under `./api`, with `./api/init` containing:
- `warehouse.csv`: initial warehouse state, i.e. list of spots with corresponding packages. Empty spot will be assign a package with id 0.
- `queue.csv` : initial queue, list of packages to drop in the warehouse. 

When the agent runs, it creates csv files at each step under `./api/sequence`, with the following format :
- `warehoue_<tag>_<step>.csv`, identical format to the initial csv file.
- `queue_<tag>_<step>.csv`, identical format to the initial csv file.

## Update scenario

To update the scenario, it is possible to :
- Change the number of steps performed, under `./api_config.json`, change `testing/nb_max_episode_steps` to given integer.
- To use the render, set `"render" : true` under `testing` of `./api_config.json`

## Advance setups

- Most constants can be found in `environmnent package/warehouse/warehouse/envs/constants/basic_constants.py`, in particular, definition of packages (sizes, frequencies), spot size (number of packages it is possible to place), warehouse filling ratio. 
- Additional constants can be found in `environmnent package`


# Training

If you want to demonstrate the software running for a longer time, I setup the config. Just :
- set `api=True`, line 35 of `environment package/warehouse/warehouse/envs/warehouse.py`. Reset to false to use the api.
- run `main.py` with python. It will load a pretrained model and run over multiple episodes.
