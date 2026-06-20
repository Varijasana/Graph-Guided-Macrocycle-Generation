# Custom Graph Transformer for Molecular Cyclisation

This repository contains a **custom-built PyTorch Graph Transformer**. It is specifically engineered to take acyclic molecule graphs (nodes and edges) and auto-regressively predict their proper sequence topologies (cyclised SMILES), following true chemical rules. 

## Capabilities
- **End-to-End Joint Training**: Automatically maps `GraphEncoder` messages to `SmilesDecoder` generations simultaneously.
- **Node-Scatter Topologies**: Replaced generic softmax implementations with rigorous node-scatter mappings (so message passing actually propagates across specific neighbors properly).
- **Inherent Physics/Rules**: Uses actual RDKit atom & bond mappings into its mathematical encodings, meaning it learns chemical affinities, not just string memorization.
- **Smart Masking**: The architecture masks dynamic padding during batching so the model never looks at fake padded atoms.

---

## How to Execute the Project 

Everything spans exactly across 3 commands. All you have to do is run them in order.

### 1) Train the Model
To start training the graph transformer from scratch:
```bash
python train.py
```
* **Auto-Resume**: If training gets canceled midway, running `python train.py` again will cleanly reload the unified checkpoint (`checkpoints/joint_checkpoint.pth`) and continue!
* It trains for 30 epochs with robust reporting metrics so you can verify improvement.

### 2) Evaluate Model Metrics
When finished training (or whenever you want to test how strong your checkpoint is):
```bash
python evaluate_model.py
```
This scores the network across a test set, outputting its actual synthetic capabilities like **Validity, Diversity, and synthetic Ring Counts (predictive cyclisation scoring)!**

### 3) Predict Single Acyclic Molecule
If someone explicitly asks you to provide an acyclic molecule SMILES and output a parsed cyclisation prediction:
```bash
python predict.py --smiles "CCC(C)C(=O)O"
```
 *(replace the SMILES string with the actual molecule string being provided).*

---

## Core Architecture
- `model.py` - The core Graph Message Passing Attention & Transformer loops.
- `decoder.py` - The sequential output generation module (with robust attention constraints).
- `train.py` - The robust End-to-End trainer.
- `predict.py` / `evaluate_model.py` - Deep testing and individual querying operations.
