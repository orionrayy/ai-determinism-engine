EndlessRPG
=========

This project is a small deterministic RPG simulation engine 
meant for experiments and reproducible research. 
It runs on Python and uses numpy pillow and torch. The code focuses on 
clear seeding checkpointing and numerical safety so runs can be reproduced exactly.

The core idea is simple. 
Encounters are generated deterministically from a seed. A compact MLP 
predicts actions and an online update loop adapts the model during play. 
The project adds guardrails like stable softmax kahan style accumulators 
and snapshot rollback so learning steps do not corrupt experiments.

To get started install dependencies then run the main script. 
A minimal example is pip install -r requirements.txt then 
python endless_rpg_v4.py or run the demo in examples. 
There is a run_unit_and_property_tests method that exercises 
numeric invariants and checkpoint integrity so add it to CI or run locally for sanity checks.

This repo is built for reproducibility and audit. 
Snapshots include canonical hashes RNG state and optional 
compressed bytes so you can verify integrity. Artifacts are written 
under artifacts_v4 and names are sanitized to avoid leaking host paths. 
The project also emits small UI snapshots and JSON metrics for easy inspection.

If you want to contribute open an issue or a pull request,
Add a short description of your change and a test when possible. 
Recommended extras include a README badge for CI 
a requirements lock file and a LICENSE such as MIT to make reuse simple.

  
*Made by github.com/orionrayy
*Instagram : @OrionRayy