## Overview

### Description
A warehouse is built in the form of a 7 x 7 grid, where each location in the grid can be used to store a package. Only one package can be stored at each location. It becomes more expensive to store and retrieve packages as we go deeper inside the grid. 

### Source
This environment corresponds to a version of Quantum Warehouse. 

## Environment
### Warehouse Grid

|  1<br>D = 0<br>R = -1 |  2<br>D = 0<br>R = -1 |  3<br>D = 0<br>R = -1 |  4<br>D = 0<br>R = -1 |  5<br>D = 0<br>R = -1 |  6<br>D = 0<br>R = -1 |  7<br>D = 0<br>R = -1 |
|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| 24<br>D = 0<br>R = -1 | 25<br>D = 1<br>R = -2 | 26<br>D = 1<br>R = -2 | 27<br>D = 1<br>R = -2 | 28<br>D = 1<br>R = -2 | 29<br>D = 1<br>R = -2 |  8<br>D = 0<br>R = -1 |
| 23<br>D = 0<br>R = -1 | 40<br>D = 1<br>R = -2 | 41<br>D = 2<br>R = -3 | 42<br>D = 2<br>R = -3 | 43<br>D = 2<br>R = -3 | 30<br>D = 1<br>R = -2 |  9<br>D = 0<br>R = -1 |
| 22<br>D = 0<br>R = -1 | 39<br>D = 1<br>R = -2 | 48<br>D = 2<br>R = -3 | 49<br>D = 3<br>R = -4 | 44<br>D = 2<br>R = -3 | 31<br>D = 1<br>R = -2 | 10<br>D = 0<br>R = -1 |
| 21<br>D = 0<br>R = -1 | 38<br>D = 1<br>R = -2 | 47<br>D = 2<br>R = -3 | 46<br>D = 2<br>R = -3 | 45<br>D = 2<br>R = -3 | 32<br>D = 1<br>R = -2 | 11<br>D = 0<br>R = -1 |
| 20<br>D = 0<br>R = -1 | 37<br>D = 1<br>R = -2 | 36<br>D = 1<br>R = -2 | 35<br>D = 1<br>R = -2 | 34<br>D = 1<br>R = -2 | 33<br>D = 1<br>R = -2 | 12<br>D = 0<br>R = -1 |
| 19<br>D = 0<br>R = -1 | 18<br>D = 0<br>R = -1 | 17<br>D = 0<br>R = -1 | 16<br>D = 0<br>R = -1 | 15<br>D = 0<br>R = -1 | 14<br>D = 0<br>R = -1 | 13<br>D = 0<br>R = -1 |

* D = Depth
* R = Reward

### Observation
Type: Box(5)

Num | Observation | Min | Max
---|---|---|---
0 | Index (Location) in Warehouse | 1 | 49
1 | Status of Occupancy | 0 (Vacant) | 1 (Occupied)
2 | Package ID | -Inf | Inf
3 | Package Weight | 0 | 1
4 | Package Type | 1 | 26

Note:
* Packages upto weight 50 kg (configurable) can be inserted. The observation returned is the normalized weight value between 0 and 1.  

### Actions
Type: Discrete(49)

Num | Action
--- | ---
1 | Insert package at location 1
2 | Insert package at location 2
.. | ..  
49 | Insert package at location 49

Note:
* The agent can only insert a package. It is withdrawn automatically by the environment.
* Only one package can be kept at a shelf.
* Every shelf has only one level. 

### Package Withdrawal Scheme
The packages are withdrawn automatically by the environment after a number of timesteps which are normally distributed for every type of package listed in the table below.

| Package Type |  µ | σ | Package Type |  µ  | σ |
|:------------:|:--:|:-:|:------------:|:---:|:-:|
|       1      |  5 | 1 |      14      |  70 | 1 |
|       2      | 10 | 1 |      15      |  75 | 1 |
|       3      | 15 | 1 |      16      |  80 | 1 |
|       4      | 20 | 1 |      17      |  85 | 1 |
|       5      | 25 | 1 |      18      |  90 | 1 |
|       6      | 30 | 1 |      19      |  95 | 1 |
|       7      | 35 | 1 |      20      | 100 | 1 |
|       8      | 40 | 1 |      21      | 105 | 1 |
|       9      | 45 | 1 |      22      | 110 | 1 |
|      10      | 50 | 1 |      23      | 115 | 1 |
|      11      | 55 | 1 |      24      | 120 | 1 |
|      12      | 60 | 1 |      25      | 125 | 1 |
|      13      | 65 | 1 |      26      | 130 | 1 |

### Reward
The reward is -1 per depth level in the warehouse grid when a package is inserted.

### Starting State
The warehouse is empty. 

### Episode Termination
* There are no more packages to insert in the warehouse.
* The warehouse is full.