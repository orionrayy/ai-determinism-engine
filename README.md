AI Determinism Engine
=====================

i made this with ai. i don’t fully know how it works but it feels stable somehow. it started as a random idea about what would happen if i tried to make a game engine that always behaves the same every time. turns out it became something like a reproducible ai sandbox.

the code handles numbers and randomness in a careful way. there are weird things inside it like kahan summation, rollback snapshots, stable softmax. i didn’t write those by hand — the ai did. i just kept testing it until it stopped breaking. everything seems deterministic now. same seed, same story, every run.

you can run it with  
`pip install -r requirements.txt`  
then  
`python ai_determinism_engine.py`  
it will create some artifacts, logs, and checks. if you run it again with the same seed, it repeats exactly. no idea why that feels so satisfying but it does.

this repo is not really about gaming. it’s more like a small proof that ai can design its own reproducible logic if you push it gently. i don’t understand the math. i just made sure it didn’t explode. somehow it didn’t.

  
*made by orionrayy*