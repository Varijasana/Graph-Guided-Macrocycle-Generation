# Graph Guided Macrocyle Generation

**GGMG** is a deep learning pipeline for macrocycle generation. By combining a **Graph Transformer** (for predicting optimal cyclisation junctions on acyclic fragments) and an **Equivariant Diffusion Model (EDM)** (for generating 3D chemical linkers), the MED pipeline constructs valid, drug-like macrocycles.

## Architecture
The pipeline consists of three core components:
1. **Graph Transformer Module:** Converts SMILES fragments into geometric graphs and uses self-attention to predict exactly where the macrocycle should close, marking junctions with `*` dummy atoms.
2. **EDM Module (DiffLinker):** A 3D equivariant diffusion model that synthesizes a chemical linker to bridge the gap between the predicted `*` junctions.
3. **Fragment-Linker Attachment & ADMET Module:** Bonds the linker to the fragment and evaluates the resulting macrocycle's physicochemical properties, outputting the Top 5 most drug-like candidates.

## How to Run on Kaggle

1. Open `med_final_updated.ipynb` in a Kaggle Notebook.
2. Upload `project.zip` as a Kaggle Dataset or directly via the **Add Data** panel in the notebook.
3. Run **Step 2A** to reference or upload the `project.zip` file.
4. Run **Step 2B** to extract and automatically configure the workspace.
5. Click **Run All** or execute the remaining cells sequentially.

> **Note:** Ensure the Kaggle notebook has internet access enabled (under **Settings → Internet**) so dependencies can be installed at runtime.

## Evaluation
Final outputs are saved as CSV files containing ADMET properties (QED, LogP, hERG risk, PAINS). High-resolution PNGs of the Top 5 generated macrocycles are saved automatically in the `intermediates/macrocycle_visuals/` directory.