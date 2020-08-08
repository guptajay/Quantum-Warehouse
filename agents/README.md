## Policies

### Random Policy
> Selects a random shelve in the warehouse.
```
# /Quantum-Warehouse
python main.py random_policy
```
![Random Policy](../img/random_policy.gif)

### Baseline Policy
> Slects the outermost shelf available in the warehouse.
```
# /Quantum-Warehouse
python main.py baseline_policy
```
![Baseline Policy](../img/baseline_policy.gif)

### Visualize with TensorBoard

```
tensorboard --logdir logs
```

### Policy Comparison (Baseline vs Random)

* **Blue** : Baseline Policy
* **Red** : Random Policy

![Policy Comparison](../img/policy_comparison.svg)

**X-Axis** : Reward, **Y-Axis** : Episode Number