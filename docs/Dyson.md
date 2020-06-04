## Required details

### Warehouse

- Shelve height of each level and number of levels. 
- Usable shelves width/depth
- Width of openings between shelves
- Time to move one/multiple shelves
- Forklift speed

### Scenario

Packages come in batches or continuous?

- Rate of retrieval; continuous or discrete ?
- Rate of droping; continuous or discrete ?
- Usual warehouse filling ratio : 

*Settled* :
 
	Usual warehouse filling ratio: 70%-80%
	Drop packages every 7 steps, retrieve packages every step. Simulates dropping every week, retieval every day

### Packages

As discussed with your mentor, packages should be gathered by spot. 

- What is the actual size of a spot. (~How many per shelve/level). : palette
- How many packages should it contain ? (~Hence, what is the package size)
- Corresponding weight of each pacakges ?
- Corresponding frequency ? 

*Setlled*

	Shelves organized with palettes, on which packages are stored.
	Need to define the differents packages types. Especially, their size, frequency and weight.

	

### Costs 

- Energy cost of moving shelves refined?
- Transport cost (forklift), marginal?



## Input/Ouput to use with physical model

**Input :** 2 csv

- Warehouse state : 

|Spot|Package|
|:---:|:--:|
|id1|classA|
|id2|-|
|id3|classC|
|id4|-|
|id5|-|

- Package to drop 

|Package|
|:--:|
|classX|

**Output :** 1 csv

|Spot|Package|
|:---:|:--:|
|id1|classA|
|id2|-|
|id3|classC|
|id4|classX|
|id4|-|

- On a separate file, the format of the different product classes are defined as a dictionnary with attributes 'size', 'frequency' and 'weight'
	
		{classA: {"frequency":.7, "weight":1, "size":1}, classB: {"frequency":.3, "weight":2, "size":3}}

## To be done

### Video simulation

After retraining on defined scenario, select sequence of dropping/retrieving packages (~15).

### Comparison with static shelves

Compare access time/ energy cost of current solution with static shelves

### Update space optimisation problem and time/cost to real life data

### Build API to run with physical model

 