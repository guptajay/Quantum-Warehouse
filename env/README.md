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
Type: Box(3)

Num | Observation | Min | Max
---|---|---|---
0 | Index (Location) in Warehouse | 1 | 49
1 | Status of Occupancy | 0 (Vacant) | 1 (Occupied)
2 | Package ID | 20 | 80

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
The packages are withdrawn automatically by the environment after a number of timesteps `WITHDRAW_TIME` (normally distributed - configurable). This package withdrawal scheme follows a normal distribution with a mean of 50 and a standard deviation of 7, both configurable.

The environment accepts packages of IDs between 20 and 80. From the distribution, we can observe that packages with IDs between 40 and 60 are sold and thus withdrawn, most often. From the figure on the left, the timesteps after which a package is withdrawn in also normally distributed. We can observe that the warehouse tries to withdraw a package often after between 4 to 6 timesteps.

![Distributions](../img/distributions.png) 

### Reward
The reward is -1 per depth level in the warehouse grid.

### Starting State
The warehouse is empty. 

### Episode Termination
There are no more packages to insert in the warehouse.