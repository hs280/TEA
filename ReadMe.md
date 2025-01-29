# README

## Repository: Supplementary Code for
**A Stochastic Techno-Economic Assessment of Emerging Artificial Photosynthetic Bio-Electrochemical Systems for CO₂ Conversion**

### Overview
This repository contains the supplementary code and data used in the paper *A Stochastic Techno-Economic Assessment of Emerging Artificial Photosynthetic Bio-Electrochemical Systems for CO₂ Conversion*. The purpose of this repository is to provide access to the computational tools and methodologies employed for conducting the stochastic techno-economic assessment (TEA) of rhodopsin-driven Artificial Photosynthetic Bioelectrochemical Systems (AP-BES).

### Abstract
Artificial Photosynthetic Bioelectrochemical Systems (AP-BES) offer a promising approach for converting CO₂ to valuable bioproducts, addressing carbon mitigation and sustainable production. This study employs a stochastic techno-economic assessment (TEA) to estimate the viability of rhodopsin-driven AP-BES, from carbon capture to product purification. Unlike traditional deterministic TEAs, this approach uses Monte Carlo simulations to model uncertainties in key technical and economic parameters, including energy consumption, CO₂ conversion efficiency, and bioproduct market prices. The analysis generates probability distributions for economic metrics such as Operational Expenditure (OPEX), Capital Expenditure (CAPEX), and profit. Enhancements in light-harvesting efficiency and advancements in reactor materials could reduce the payback period to just one year, thereby making large-scale deployment a feasible option.

### Repository Structure
```
├── Bin/        # Contains Python scripts and Jupyter notebooks for TEA analysis
├── Test/       # Contains Code to generate plots for each subsystem 
├── additional_supplementary/  # Additional supplementary files
│   ├── S1_Supplementary_Results.docx    # Detailed supplementary results
│   ├── S2_Supplementary_Methodology.docx    # Additional methods description
├── README.md                  # This file
```

### Dependencies
To run the code provided in this repository, the following dependencies are required:
- Python 3.11+
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- plotly



### Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-link.git
   cd your-repo-link
   ```
2. Navigate to the `Test` folder.
3. Run the provided scripts to reproduce the stochastic TEA analysis.
4. For additional details on methodology and results, refer to `additional_supplementary/S2_Supplementary_Methods.pdf` and `additional_supplementary/S1_Supplementary_Results.pdf`.

### Citation
If you use this repository in your research, please cite:
**[Full Paper Citation Here]**

### Contact
For any queries or further information, please contact the corresponding author at **haris.saeed@eng.ox.ac.uk**.

---

This repository aims to facilitate reproducibility and further exploration of stochastic TEA for AP-BES development. We welcome contributions and discussions!

