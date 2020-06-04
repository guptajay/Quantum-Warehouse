# Computing the acces times and energy costs

## Setting

1 spot is a square of 0.5x0.5. This allows for a level to have a width of 1.

n_layers = 3 --> max_level = 6

(level,orientation,position) --> ((levels to be moved in orientation 0|1), unit diagonal drive, unit aisle drive)
(6,0,i) --> ( (-|-), 0, i)
(6,1,i) --> ( (-|-), 0, i+1)
(5,0,i) --> ( (6,5|-),1, i-1)
(5,1,i) --> ( (6,5|6,5),1, i)
(4,0,i) --> ( (6,5|-),1, i)
(4,1,i) --> ( (6,5|6,5),1,i+1)
(3,0,i) --> ( (6,5,4,3|6,5), 2,i-1)
(3,1,i) --> ( (6,5,4,3|6,5,4,3), 2,i)
(2,0,i) --> ( (6,5,4,3|6,5), 2,i)
(2,1,i) --> ( (6,5,4,3|6,5,4,3), 2,i+1)
(1,0,i) --> ( (6,5,4,3,2,1|6,5,4,3), 3,i-1)
(1,1,i) --> ( (6,5,4,3,2,1|6,5,4,3,2,1), 3,i)
(0,0,1) --> ( (6,5,4,3,2,1|6,5,4,3), 3,1)
(0,1,1) --> ( (6,5,4,3,2,1|6,5,4,3,2,1), 3,i)

## computation

### Access time

#### Orientation 0

access_time = time_to_open shelve (if needed) + time to drive diagonnally + time to drive in the spot
```access_time(level,position, max_level) = time_to_open shelve * ( 1 if level < max_level else 0 ) + unit_diagonal_drive_time * (max_level-level) + ( position - (1 if level is odd else 0) )*unit_drive_time/2```

#### Orientation 1

access_time = time_to_open shelve (if needed) + time to drive diagonnally + time to drive in the spot
```access_time(level,position, max_level) = time_to_open shelve * ( 1 if level < max_level else 0 ) + unit_diagonal_drive_time * (max_level-level) + ( position + (1 if level is even else 0) )*unit_drive_time/2```

### Emergy cost

energy_cost = total weight to move  x  units of distance to move  x unit cost of moving one unit weight for one unit distance

#### Orientation 0

```energy_cost(level, max_level) = [ sum( weight for weight in layer for layer in layers of orientation 0 whose level >= closest superior odd level) +  sum( weight for weight in layer for layer in layers of orientation 0 whose level >= closest superior odd level + 2)  ] * 1 * unit_weight_displacement_cost```

#### Orientation 1

```energy_cost(level, max_level) = [ sum( weight for weight in layer for layer in layers of orientation 0 whose level >= closest superior odd level) +  sum( weight for weight in layer for layer in layers of orientation 0 whose level >= closest superior odd level)  ] * 1 * unit_weight_displacement_cost```
