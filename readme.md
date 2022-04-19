# How Gender Debiasing Affects Internal Model Representations, and Why It Matters 

This repo contains the code for reproducing the results in the paper.

The repo is organized as follows:

1. **bio** - contains the code to train and test the models on Bias in Bios task.

2. **coref** - contains the code to train and test the models on coreference resolution task (Ontonotes and Winobias).

3. **compression** - contains the code to measure the compression rate of gender information in a model, using MDL probes.

4. **CEAT** - contains the code to measure the metric CEAT in models.

Each folder contains its own documentation, and we provide links to the data (or ways to reproduce it) and links to model checkpoints.
The code of the experiments contains code to automatically log the results to [Wandb](https://wandb.ai/).

For questions, please reach out to the authors of the paper.