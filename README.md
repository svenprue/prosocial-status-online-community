# The Complementarity of Prosocial and Status-Seeking Behavior in Contributor Recruitment and Retention in Online Communities

This repository contains the full code to run create the datasets, run the statistical models, and reproduce all figures for the article:

**Strahringer, L., Prüß, S.: _The Complementarity of Prosocial and Status-Seeking Behavior in Contributor Recruitment and Retention in Online Communities_**

## Contents
- **main.Rmd** - The main R Markdown file to execute the analysis  
- **01_input_data/** - Folder containing input data files  
- **02_output_plots/** - Folder for storing simulation results and generated plots  
- **03_functions/** - Helper functions for data processing and analysis  
- **04_plotting/** - Plotting functions for generating figures  

## Instructions

There are two options to use this code, controlled by a switch in `main.Rmd`:

### **1. Reproduction Mode (Default)**
- Download the pre-processed data files (if available) and place them into `01_input_data/`
- Run `main.Rmd`
- This will import the data, reproduce all figures from the article, and save them into `02_output_plots/`

### **2. Simulation Mode**
- Set `model.mode = "simulation"` in `main.Rmd`
- Adjust parameters such as the sample size `n` for simulations as needed
- Run `main.Rmd`
- This will initiate the full simulation, which may take some time depending on the sample size
- The output figures will be saved in `02_output_plots/`. Note that results might slightly deviate from the article figures due to randomness in the simulation process.

## Code Environment

Make sure the following R packages are installed before running the code:

```r
install.packages(c("tidyverse", "data.table", "ggplot2", "lme4", "broom", "sandwich", "lmtest", "stargazer"))
