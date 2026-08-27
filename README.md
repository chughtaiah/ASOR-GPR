Reproducibility Code for Variational Outlier-Robust GPR

This repository contains the MATLAB experiments accompanying “Variational Outlier-Robust Gaussian Process Regression with Generative Modeling” (arXiv:2608.16606). The scripts compare Oracle, ASOR-GPR, GMM-GPR, Student-(t) GPR, RCGPR, and standard GPR under synthetic and real-data contamination.

Requirements

MATLAB R2020a or newer is recommended for running the scripts unchanged; they use local functions, boxchart, and exportgraphics.

GPML MATLAB v4.2 (2018-06-11) is required for the Student-(t) GPR baseline.

Parallel Computing Toolbox is required by all Monte Carlo sweep scripts. The timing script runs serially by design.

Enough memory for one complete model fit per parallel worker. If MATLAB runs out of memory, reduce numWorkers before reducing the experimental settings.

GPML is third-party software and should be downloaded separately under its own license.

Recommended repository layout

.
├── README.md
├── code/
│   ├── All_Methods_Timing_Simulation.m
│   ├── Out_prob_sweep_sup_results.m
│   ├── Outlier_Probability_Sweep_Simulation.m
│   ├── Real_Data_Simulation_1.m
│   ├── Real_Data_Simulation_2.m
│   └── Real_Data_Supplementary_Results.m
├── data/
│   ├── AirQualityUCI_clean_numeric.csv
│   └── energy_efficiency_clean_numeric_confirmed.csv
├── external/
│   └── gpml-matlab-v4.2-2018-06-11/
└── results/                         # Created by the scripts

One-time setup

Download GPML v4.2 and place it in external/gpml-matlab-v4.2-2018-06-11, or use another local location.

Place the two cleaned, numeric real-data CSV files in data/. Each file must contain at least 200 complete rows.

Open each script you intend to run and replace its Windows-specific paths near the top:

gpml_path or gpmlPath: full path to the GPML directory.

userRoot, outputRoot, or outputRootDir: desired results directory.

airQualityDataPath and energyDataPath: full paths to the two CSV files.

experiments(1).realDataPath: Air Quality CSV path in Real_Data_Simulation_1.m.

A portable pattern is:

repoRoot = fileparts(fileparts(mfilename('fullpath'))); % if scripts are in code/
gpmlPath = fullfile(repoRoot, 'external', ...
    'gpml-matlab-v4.2-2018-06-11');
dataDir = fullfile(repoRoot, 'data');
outputRootDir = fullfile(repoRoot, 'results');

airQualityDataPath = fullfile(dataDir, ...
    'AirQualityUCI_clean_numeric.csv');
energyDataPath = fullfile(dataDir, ...
    'energy_efficiency_clean_numeric_confirmed.csv');

Use the variable name already present in each script (gpml_path versus gpmlPath). Do not add GPML to the repository unless its license permits redistribution.

Data conventions

Real_Data_Supplementary_Results.m removes rows containing non-finite values, then uses:

the last three columns of the Air Quality CSV as outputs;

the last two columns of the Energy Efficiency CSV as outputs; and

all preceding columns as inputs.

The code standardizes inputs and outputs using training-set statistics only. Preserve the same numeric column ordering when regenerating the cleaned CSV files.

Quick smoke test

The paper settings are deliberately expensive. Before a full run, temporarily use:

nMC = 2;
pOutList = [0 0.4];
numWorkers = 2;

For All_Methods_Timing_Simulation.m, temporarily use:

nTrainList = [200 400];
nMC = 2;

Confirm that all methods complete and that the .mat, .csv, .png, .eps, and .fig outputs are produced. Restore the original settings before generating manuscript results.

Running the experiments

Start MATLAB in the repository root and run each script independently:

cd code

run('Outlier_Probability_Sweep_Simulation.m')
run('Out_prob_sweep_sup_results.m')
run('Real_Data_Supplementary_Results.m')
run('All_Methods_Timing_Simulation.m')

Run the timing experiment last. It closes an existing parallel pool and restricts MATLAB to one computational thread so that fitting times are measured serially.

The following scripts are optional isolated cases and are not required for the current supplementary composite figures:

run('Real_Data_Simulation_1.m')
run('Real_Data_Simulation_2.m')

Script guide

Script

Purpose

Full settings

Principal output

Outlier_Probability_Sweep_Simulation.m

Main synthetic outlier-probability sweep

(p_{\mathrm{out}}=0:0.1:0.8), 30 Monte Carlo trials, 100 training and 100 test samples

RMSE boxplot, RMSE/convergence CSVs, and results MAT file

Out_prob_sweep_sup_results.m

Supplementary synthetic sweeps

Gaussian variance 10 and 50; uniform ([0,15]), ([-15,0]), and ([-10,10])

One result folder and RMSE figure per contamination model, plus a manifest MAT file

Real_Data_Supplementary_Results.m

Eight real-data cases

Air Quality and Energy Efficiency, each with four contamination models

One folder per case, checkpoint files, figures, CSVs, and a master MAT file

All_Methods_Timing_Simulation.m

Serial fitting-time scaling

(n_{\mathrm{train}}\in{200,400,600,800,1000}), 30 trials, (p_{\mathrm{out}}=0.4)

Linear/log runtime boxplots, log–log plot, iteration plot, CSV/TXT summaries, and MAT file

Real_Data_Simulation_1.m

Optional Air Quality diagnostic

Uniform ([-5,15]) contamination

Single-case results and diagnostic outputs

Real_Data_Simulation_2.m

Optional Energy Efficiency diagnostic

Gaussian contamination with variance 25

Single-case results and RMSE figure

Reproducing the supplementary figures

The current synthetic composite contains one Gaussian case and three uniform cases. After running Out_prob_sweep_sup_results.m, use the following EPS files:

Supplementary panel

Generated file

Suggested manuscript filename

Gaussian, variance 10

01_Gaussian_Var10/Gaussian_Var10_pout_sweep_rmse_boxplot.eps

G_10.eps

Uniform ([-10,10])

05_Uniform_neg10_10/Uniform_neg10_10_pout_sweep_rmse_boxplot.eps

U_neg10_pos10.eps

Uniform ([0,15])

03_Uniform_0_15/Uniform_0_15_pout_sweep_rmse_boxplot.eps

U_P_15.eps

Uniform ([-15,0])

04_Uniform_neg15_0/Uniform_neg15_0_pout_sweep_rmse_boxplot.eps

U_neg_15.eps

The Gaussian-variance-50 case is an additional robustness check and can be omitted from the four-panel figure.

After running Real_Data_Supplementary_Results.m, the eight case folders map to the manuscript files as follows:

Case folder

Generated EPS file

Suggested manuscript filename

01_AirQuality_Uniform_neg5_5

AirQuality_Uniform_neg5_5_pout_sweep_boxplot.eps

AQ_U_neg5_5pos.eps

02_AirQuality_Gaussian_Var10

AirQuality_Gaussian_Var10_pout_sweep_boxplot.eps

AQ_G_10.eps

03_AirQuality_Uniform_0_15

AirQuality_Uniform_0_15_pout_sweep_boxplot.eps

AQ_U_P_15.eps

04_AirQuality_Uniform_neg15_0

AirQuality_Uniform_neg15_0_pout_sweep_boxplot.eps

AQ_U_N_15.eps

05_EnergyEfficiency_Uniform_neg5_5

EnergyEfficiency_Uniform_neg5_5_pout_sweep_boxplot.eps

EE_U_neg5_pos5.eps

06_EnergyEfficiency_Gaussian_Var10

EnergyEfficiency_Gaussian_Var10_pout_sweep_boxplot.eps

EE_G_10.eps

07_EnergyEfficiency_Uniform_0_15

EnergyEfficiency_Uniform_0_15_pout_sweep_boxplot.eps

EE_U_P_15.eps

08_EnergyEfficiency_Uniform_neg15_0

EnergyEfficiency_Uniform_neg15_0_pout_sweep_boxplot.eps

EE_U_N_15.eps

For the runtime figure, copy or rename:

All_methods_fit_time_boxplot_linear.eps  ->  Extensive_timing.eps

The scripts retain their original generated filenames; the right-hand names above are only the names expected by the manuscript LaTeX source.

Reproducibility notes

The scripts initialize the random-number generator and store settings with the numerical results.

Full sweeps use 30 Monte Carlo trials and save intermediate checkpoints where implemented.

Non-Oracle methods receive only the contaminated training data. The Oracle alone receives the realized contamination labels and specified component variances.

The timing script measures model fitting only; data generation, prediction, and plotting are outside the timed section.

Output directories can be large because figures are saved in several formats and complete Monte Carlo arrays are retained.

Troubleshooting

startup.m or GPML functions cannot be found. Check gpml_path/gpmlPath, then run GPML's startup.m once manually to confirm the installation.

The parallel pool fails or MATLAB runs out of memory. Reduce numWorkers to 2 or 4. Each worker fits its own set of GP models and therefore needs substantial memory.

The real-data script reports too few valid rows. Verify that each cleaned CSV has at least 200 fully numeric, finite rows and preserves the expected output columns at the end.

EPS export is unavailable. Use MATLAB R2020a or newer. The scripts also save PNG and FIG files, but the manuscript workflow expects EPS.

A full run appears stalled. The robust methods perform repeated latent-variable inference and kernel-hyperparameter optimization. Full sweeps—especially the serial timing experiment—can take many hours.

Citation

If you use this code, please cite:

@misc{majal2026variational,
  title         = {Variational Outlier-Robust Gaussian Process Regression
                   with Generative Modeling},
  author        = {Arslan Majal and Aamir Hussain Chughtai},
  year          = {2026},
  eprint        = {2608.16606},
  archivePrefix = {arXiv}
}

Please open a GitHub issue if a run fails after the paths and dependencies have been checked.
