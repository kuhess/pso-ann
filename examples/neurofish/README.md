Install the dependencies

```
poetry install
```

## Train

```
poetry run python train.py
```

This will generate two files:
- the first `[DATETIME]_dummy_particle.pickle` before the training
- the second `[DATETIME]_best_particle.pickle` with the weight of the best particle after the training


## Run

Run a fish without optimization (random weights)
```
poetry run python main.py dummy.pickle
```

Run a trained fish
```
poetry run python main.py best.pickle
```

*Note: you can add a new reward by clicking in the window.*
