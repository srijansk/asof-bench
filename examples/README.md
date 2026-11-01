# examples

[toy_world.py](toy_world.py) builds a toy world to develop a system against: three small synthetic repositories laid out like python/peps, ethereum/EIPs and ethereum/ERCs, on which it runs every command. `build` writes the toy questions and gold, `stream` the H- stream, `materialize` the files and log that grep at D reads on one date, and `score` scores an answers file of one constant answer per item for an outside entry, `my-system`.

```bash
uv run python examples/toy_world.py ../asofbench-toy
```

It takes about two seconds and prints where it wrote each part. Replace the constant answers with your system's to check your answer rows, your entries file and the scorer's report before any real run. The registration forbids a configuration to answer a v0.1 item before the commit that fixes it ([section 11](../PREREGISTRATION.md#11-systems-model-and-runs)): develop here, never on data/v0.1. None of the toy items is a v0.1 item, and a toy run's results say nothing about v0.1. [docs/entering-a-system.md](../docs/entering-a-system.md) has the rules a real entry follows, and [docs/data-format.md](../docs/data-format.md) the format of every file.
