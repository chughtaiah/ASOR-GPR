%% RealData_Extensive
%
% Runs eight real-data robustness experiments:
%   Air Quality and Energy Efficiency, each with
%       1. U[-5,5]
%       2. N(0,10), where 10 is the variance
%       3. U[0,15]
%       4. U[-15,0]
%
% Protocol:
%   1. Randomly select 100 training and 100 test observations.
%   2. Standardize X and Y using training-set statistics only.
%   3. Add entry-wise artificial contamination only to standardized Ytrain.
%   4. Add no extra nominal noise to either real dataset.
%   5. Evaluate against the untouched standardized Ytest.
%   6. Sweep p_out = 0,0.1,...,0.8 over 30 Monte Carlo trials.
%   7. Save a checkpoint after every completed p_out value.
%   8. Plot and save each of the eight figures immediately when its full
%      p_out sweep is complete, before advancing to the next case.
%
% Methods:
%   1. Oracle (known entry-wise labels and specified component variances)
%   2. ASOR-GPR
%   3. GMM-GPR
%   4. Student-t GPR (GPML)
%   5. RCGPR
%   6. Standard GPR
%
% STRICT INFORMATION BOUNDARY:
%   - Only Oracle receives the generated contamination mask, the generating
%     distribution parameters, or the corresponding component variance.
%   - ASOR-GPR, GMM-GPR, Student-t GPR, RCGPR, and Standard GPR receive only
%     Xtrain, contaminated Ytrain, and fixed contamination-independent
%     settings.
%   - RCGPR epsilon is fixed at 0.05 for every p_out and every outlier case.
%   - GMM initialization multipliers are fixed at 10 and 1 for every case;
%     they are internal algorithm settings, not oracle information.

clear; clc; close all;
rng(7, 'twister');

%% ========================================================================
% User paths
% ========================================================================
airQualityDataPath = "C:\Users\majal\Downloads\GPR_Datasets\GPR_Datasets\AirQualityUCI_clean_numeric.csv";
energyDataPath = "C:\Users\majal\Downloads\GPR_Datasets\GPR_Datasets\energy_efficiency_clean_numeric_confirmed.csv";
gpmlPath = 'C:\Users\majal\Downloads\gpml-matlab-v4.2-2018-06-11';

scriptDir = fileparts(mfilename('fullpath'));
outputRootDir = fullfile(scriptDir, 'real_data_extensive_8_cases_results');

if ~exist(airQualityDataPath, 'file')
    error('Air Quality CSV file not found: %s', airQualityDataPath);
end

if ~exist(energyDataPath, 'file')
    error('Energy Efficiency CSV file not found: %s', energyDataPath);
end

if ~exist(gpmlPath, 'dir')
    error('GPML directory not found: %s', gpmlPath);
end

if ~exist(outputRootDir, 'dir')
    mkdir(outputRootDir);
end

%% ========================================================================
% GPML and parallel setup
% ========================================================================
addpath(genpath(gpmlPath));
evalc('startup;');

numWorkers = 12;

if license('test', 'Distrib_Computing_Toolbox') ~= 1
    error('Parallel Computing Toolbox is required because MC trials use parfor.');
end

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', numWorkers);
else
    fprintf('Using existing parallel pool with %d workers.\n', pool.NumWorkers);
end

% Make GPML available on every worker.
future = parfevalOnAll(@addpath, 0, genpath(gpmlPath));
wait(future);
future = parfevalOnAll(@() evalc('startup;'), 0);
wait(future);

% Prevent nested BLAS threading from oversubscribing the CPU.
try
    future = parfevalOnAll(@maxNumCompThreads, 0, 1);
    wait(future);
catch ME
    warning('Could not limit worker BLAS threads: %s', ME.message);
end

%% ========================================================================
% Experiment configuration
% ========================================================================
nTrain = 100;
nTest = 100;
nMC = 30;

pOutList = 0:0.1:0.8;
nPout = numel(pOutList);

% Common initialization supplied to every non-oracle method. It is fixed
% before the experiments and never depends on p_out or the outlier case.
sigmaInit = 0.5;

% The real datasets have unknown inherent measurement variance. The Oracle
% uses the same assumed inlier variance scale as the common initialization.
% Its additional information is confined to its private branch below.
oracleInlierVariance = sigmaInit^2;

% Dataset metadata. Empty outputColumns means the last numOutputs columns.
datasets = struct([]);

datasets(1).name = 'Air Quality';
datasets(1).tag = 'AirQuality';
datasets(1).dataPath = airQualityDataPath;
datasets(1).outputColumns = [];
datasets(1).numOutputs = 3;

datasets(2).name = 'Energy Efficiency';
datasets(2).tag = 'EnergyEfficiency';
datasets(2).dataPath = energyDataPath;
datasets(2).outputColumns = [];
datasets(2).numOutputs = 2;

% Outlier metadata. For Gaussian contamination, scale is the standard
% deviation, hence sqrt(10) gives variance 10.
outlierCases = struct( ...
    'tag', { ...
        'Uniform_neg5_5', ...
        'Gaussian_Var10', ...
        'Uniform_0_15', ...
        'Uniform_neg15_0'}, ...
    'label', { ...
        'U[-5,5]', ...
        'N(0,10)', ...
        'U[0,15]', ...
        'U[-15,0]'}, ...
    'model', { ...
        'shifted_uniform', ...
        'gaussian', ...
        'shifted_uniform', ...
        'shifted_uniform'}, ...
    'scale', { ...
        [-5,5], ...
        sqrt(10), ...
        [0,15], ...
        [-15,0]});

nDatasets = numel(datasets);
nOutlierCases = numel(outlierCases);
nExperiments = nDatasets * nOutlierCases;

%% ========================================================================
% Shared model settings
% ========================================================================
opts = struct();

% ASOR/GMM outer-loop convergence.
opts.maxIter = 1000;
opts.tol = 1e-5;
opts.outerConvergenceCriterion = 'obj_window3_max';

% Shared ARD-kernel initialization and learning.
opts.init_signal_std = 1.0;
opts.init_lengthscale = 1.0;
opts.learnTheta = true;
opts.thetaBurnIn = 0;
opts.thetaUpdateEvery = 1;

% Warm-started analytic-gradient Armijo descent.
opts.thetaOptimizer = 'gd_warmstart';
opts.thetaGradStep = 0.05;
opts.thetaGradMaxIter = 25;
opts.thetaMaxIter = opts.thetaGradMaxIter;
opts.thetaGradArmijo = 1e-4;
opts.thetaGradMinStep = 1e-7;
opts.thetaGradVerbose = false;
opts.thetaInnerRelTol = 1e-5;
opts.thetaGradTol = 1e-5;
opts.thetaMinInnerIter = 2;
opts.thetaStepIncrease = 1.25;
opts.thetaGradStepMinStart = 1e-5;
opts.thetaGradStepMaxStart = opts.thetaGradStep;

% One-shot optimization budget for Standard GPR and Oracle.
opts.fixedGPGradMaxIter = 60;

% ASOR priors.
opts.a = 1;
opts.theta0 = 0.5;
opts.A = 10;
opts.nu0_base_offset = 2;

nu0Scalar = 1 + opts.nu0_base_offset;
opts.S0_scale = (nu0Scalar + 2) * sigmaInit^2;

robustScaleFactor = 20;
opts.init_b = opts.a * robustScaleFactor;
opts.B = (opts.A - 1) / opts.init_b;

% GMM-specific initialization. Component 1 is the broad/outlier component.
opts.gmmOutlierInitMultiplier = 10;
opts.gmmNominalInitMultiplier = 1;

% RCGPR settings.
% Fixed a priori for all p_out values; never set this from pOutNow.
fixedRcgprEpsilon = 0.05;
opts.rcgp_epsilon = fixedRcgprEpsilon;
opts.rcgp_useShrinkageTerm = true;
opts.rcgp_shrinkageConvention = 'section3';
opts.rcgp_maxIter = 100;
opts.rcgp_lbfgsMemory = 10;
opts.rcgp_gradTol = 1e-5;
opts.rcgp_stepTol = 1e-10;
opts.rcgp_armijo = 1e-4;
opts.rcgp_minStep = 1e-10;
opts.rcgp_verbose = false;

% Student-t GPML settings.
opts.student_gpml_inference = 'VB';
opts.student_gpml_nu = 4;
opts.student_gpml_optIters = -60;

opts.jitter = 1e-6;
opts.minWeight = 1e-6;
opts.verbose = false;

%% ========================================================================
% Method metadata and result arrays
% ========================================================================
methodNames = {'Oracle', 'ASOR-GPR', 'GMM-GPR', ...
               'Student-t GPR', 'RCGPR', 'Standard GPR'};

colors = [
    0.25 0.25 0.25;
    35 139 69;
    180 60 120;
    0.75*255 0.10*255 0.10*255;
    20 120 180;
    220 120 20
] / 255;

pOutLabels = strings(1, nPout);
for pp = 1:nPout
    pOutLabels(pp) = sprintf('%.1f', pOutList(pp));
end

allResults = struct([]);
totalWallTimer = tic;
experimentIndex = 0;

fprintf('\n============================================================\n');
fprintf('REAL-DATA EXTENSIVE SWEEP: %d datasets x %d cases = %d figures\n', ...
    nDatasets, nOutlierCases, nExperiments);
fprintf('nTrain = %d | nTest = %d | nMC = %d\n', nTrain, nTest, nMC);
fprintf('p_out = %s\n', mat2str(pOutList));
fprintf('Fixed non-oracle sigma initialization = %.3f\n', sigmaInit);
fprintf('Fixed RCGPR epsilon = %.3f\n', fixedRcgprEpsilon);
fprintf('Fixed GMM multipliers: broad = %.1f R0, nominal = %.1f R0\n', ...
    opts.gmmOutlierInitMultiplier, opts.gmmNominalInitMultiplier);
fprintf('Root results directory: %s\n', outputRootDir);
fprintf('============================================================\n');

for datasetIdx = 1:nDatasets
for outlierCaseIdx = 1:nOutlierCases

experimentIndex = experimentIndex + 1;
datasetCfg = datasets(datasetIdx);
outlierCfg = outlierCases(outlierCaseIdx);

caseName = sprintf('%s_%s', datasetCfg.tag, outlierCfg.tag);
caseOutputDir = fullfile(outputRootDir, sprintf('%02d_%s', experimentIndex, caseName));
if ~exist(caseOutputDir, 'dir')
    mkdir(caseOutputDir);
end

[outlierVariance, outlierSecondMoment, outlierMean] = ...
    outlier_stats_local(outlierCfg.model, outlierCfg.scale);

% Oracle-only broad component. This quantity is never inserted into optsRun.
oracleOutlierVariance = oracleInlierVariance + outlierSecondMoment;

fprintf('\n\n============================================================\n');
fprintf('Starting experiment %d/%d: %s\n', experimentIndex, nExperiments, caseName);
fprintf('Dataset = %s\n', datasetCfg.name);
fprintf('Contamination = %s\n', outlierCfg.label);
fprintf('Generator model/scale = %s / %s\n', ...
    outlierCfg.model, mat2str(outlierCfg.scale));
fprintf('Generator variance = %.6f | mean = %.6f | E[eta^2] = %.6f\n', ...
    outlierVariance, outlierMean, outlierSecondMoment);
fprintf('Case directory: %s\n', caseOutputDir);
fprintf('============================================================\n');

RMSE_ORACLE = nan(nMC, nPout);
RMSE_ASOR = nan(nMC, nPout);
RMSE_GMM = nan(nMC, nPout);
RMSE_STUDENT = nan(nMC, nPout);
RMSE_RCGP = nan(nMC, nPout);
RMSE_STD = nan(nMC, nPout);

fitTime_ASOR = nan(nMC, nPout);
fitTime_GMM = nan(nMC, nPout);

ASOR_outerConverged = false(nMC, nPout);
GMM_outerConverged = false(nMC, nPout);
ASOR_hitMaxIter = false(nMC, nPout);
GMM_hitMaxIter = false(nMC, nPout);
ASOR_iterMedian = nan(nMC, nPout);
GMM_iterMedian = nan(nMC, nPout);
ASOR_iterMax = nan(nMC, nPout);
GMM_iterMax = nan(nMC, nPout);
ASOR_finalRelMax = nan(nMC, nPout);
GMM_finalRelMax = nan(nMC, nPout);
ASOR_innerConvFrac = nan(nMC, nPout);
GMM_innerConvFrac = nan(nMC, nPout);
ASOR_innerHitCapFrac = nan(nMC, nPout);
GMM_innerHitCapFrac = nan(nMC, nPout);
ASOR_lineSearchFailFrac = nan(nMC, nPout);
GMM_lineSearchFailFrac = nan(nMC, nPout);

outlierEntryCount = zeros(nMC, nPout);
outlierRowCount = zeros(nMC, nPout);
inputDimension = zeros(nMC, nPout);
outputDimension = zeros(nMC, nPout);

caseWallTimer = tic;

%% ========================================================================
% Monte Carlo experiment
% ========================================================================
for pp = 1:nPout

    pOutNow = pOutList(pp);
    pOutTimer = tic;

    fprintf('\n%s | starting p_out = %.2f (%d/%d)\n', ...
        caseName, pOutNow, pp, nPout);

    parfor mc = 1:nMC

        seedNow = 1000000 + 100000*datasetIdx + ...
            10000*outlierCaseIdx + 100*pp + mc;
        rng(seedNow, 'twister');

        [Xtrain, Xtest, YcleanTrain, YcleanTest] = load_real_numeric_split_csv( ...
            datasetCfg.dataPath, nTrain, nTest, seedNow, ...
            datasetCfg.outputColumns, datasetCfg.numOutputs);

        d_x = size(Xtrain, 2);
        p_y = size(YcleanTrain, 2);

        inputDimension(mc, pp) = d_x;
        outputDimension(mc, pp) = p_y;

        optsRun = opts;
        optsRun.nu0 = p_y + opts.nu0_base_offset;
        optsRun.init_R = sigmaInit^2 * eye(p_y);
        optsRun.sigma_init = sigmaInit;
        optsRun.rcgp_epsilon = fixedRcgprEpsilon;

        ell0 = 0.20 * median_pairwise_distance(Xtrain);
        if ~isfinite(ell0) || ell0 <= 0
            ell0 = opts.init_lengthscale;
        end
        optsRun.init_lengthscale = max(ell0, 1e-2);
        optsRun.init_signal_std = 1.0;

        % Fail immediately if experiment truth enters non-oracle options.
        assert_nonoracle_information_boundary( ...
            optsRun, [datasetCfg.name ' non-oracle methods'], fixedRcgprEpsilon);

        % The generator alone uses pOutNow and the case distribution.
        % No non-oracle method receives either quantity or the mask.
        outlierNoise = generate_outlier_noise_local( ...
            nTrain, p_y, outlierCfg.model, outlierCfg.scale);
        isOutlierEntry = rand(nTrain, p_y) < pOutNow;

        Ytrain = YcleanTrain;
        Ytrain(isOutlierEntry) = Ytrain(isOutlierEntry) ...
            + outlierNoise(isOutlierEntry);

        outlierEntryCount(mc, pp) = nnz(isOutlierEntry);
        outlierRowCount(mc, pp) = nnz(any(isOutlierEntry, 2));

        % ASOR-GPR.
        fitTimer = tic;
        modelASOR = independent_asor_gpr_fit_lrdiag(Xtrain, Ytrain, optsRun);
        fitTime_ASOR(mc, pp) = toc(fitTimer);
        Ypred = independent_asor_gpr_predict_lrdiag(modelASOR, Xtest);
        RMSE_ASOR(mc, pp) = sqrt(mean((Ypred(:) - YcleanTest(:)).^2));

        diagASOR = collect_independent_fit_diagnostics_cap(modelASOR);
        ASOR_outerConverged(mc, pp) = diagASOR.allConverged;
        ASOR_hitMaxIter(mc, pp) = diagASOR.anyHitMaxIter;
        ASOR_iterMedian(mc, pp) = diagASOR.iterMedian;
        ASOR_iterMax(mc, pp) = diagASOR.iterMax;
        ASOR_finalRelMax(mc, pp) = diagASOR.finalRelMax;
        ASOR_innerConvFrac(mc, pp) = diagASOR.thetaInnerConvFrac;
        ASOR_innerHitCapFrac(mc, pp) = diagASOR.thetaInnerHitCapFrac;
        ASOR_lineSearchFailFrac(mc, pp) = diagASOR.thetaLineSearchFailFrac;

        % GMM-GPR.
        fitTimer = tic;
        modelGMM = independent_gmm_gpr_fit_lrdiag(Xtrain, Ytrain, optsRun);
        fitTime_GMM(mc, pp) = toc(fitTimer);
        Ypred = independent_gmm_gpr_predict_lrdiag(modelGMM, Xtest);
        RMSE_GMM(mc, pp) = sqrt(mean((Ypred(:) - YcleanTest(:)).^2));

        diagGMM = collect_independent_fit_diagnostics_cap(modelGMM);
        GMM_outerConverged(mc, pp) = diagGMM.allConverged;
        GMM_hitMaxIter(mc, pp) = diagGMM.anyHitMaxIter;
        GMM_iterMedian(mc, pp) = diagGMM.iterMedian;
        GMM_iterMax(mc, pp) = diagGMM.iterMax;
        GMM_finalRelMax(mc, pp) = diagGMM.finalRelMax;
        GMM_innerConvFrac(mc, pp) = diagGMM.thetaInnerConvFrac;
        GMM_innerHitCapFrac(mc, pp) = diagGMM.thetaInnerHitCapFrac;
        GMM_lineSearchFailFrac(mc, pp) = diagGMM.thetaLineSearchFailFrac;

        % Student-t GPR.
        modelStudent = studentt_gpml_mogp_fit(Xtrain, Ytrain, optsRun);
        Ypred = studentt_gpml_mogp_predict(modelStudent, Xtest);
        RMSE_STUDENT(mc, pp) = sqrt(mean((Ypred(:) - YcleanTest(:)).^2));

        % RCGPR.
        modelRCGP = independent_rcgp_gpr_fit(Xtrain, Ytrain, optsRun);
        Ypred = independent_rcgp_gpr_predict(modelRCGP, Xtest);
        RMSE_RCGP(mc, pp) = sqrt(mean((Ypred(:) - YcleanTest(:)).^2));

        % Standard GPR.
        modelStandard = independent_standard_gpr_fit(Xtrain, Ytrain, optsRun);
        Ypred = independent_standard_gpr_predict(modelStandard, Xtest);
        RMSE_STD(mc, pp) = sqrt(mean((Ypred(:) - YcleanTest(:)).^2));

        % Oracle with known entry labels and specified component variances.
        optsOracle = optsRun;
        optsOracle.oracle_inlier_var = oracleInlierVariance;
        optsOracle.oracle_outlier_var = oracleOutlierVariance;

        modelOracle = independent_oracle_gmm_style_gpr_fit( ...
            Xtrain, Ytrain, isOutlierEntry, optsOracle);
        Ypred = independent_oracle_gmm_style_gpr_predict(modelOracle, Xtest);
        RMSE_ORACLE(mc, pp) = sqrt(mean((Ypred(:) - YcleanTest(:)).^2));

        fprintf(['%s | p_out %.2f | MC %2d/%2d | Oracle %.4f | ASOR %.4f | ', ...
            'GMM %.4f | Student-t %.4f | RCGPR %.4f | Standard %.4f\n'], ...
            caseName, pOutNow, mc, nMC, ...
            RMSE_ORACLE(mc, pp), RMSE_ASOR(mc, pp), ...
            RMSE_GMM(mc, pp), RMSE_STUDENT(mc, pp), ...
            RMSE_RCGP(mc, pp), RMSE_STD(mc, pp));
    end

    fprintf('%s | completed p_out = %.2f in %.2f minutes.\n', ...
        caseName, pOutNow, toc(pOutTimer)/60);

    % Save immediately after every completed probability level so a long
    % run can be inspected or resumed manually without losing prior levels.
    checkpointFile = fullfile(caseOutputDir, sprintf( ...
        '%s_checkpoint_pout_%02d.mat', caseName, round(100*pOutNow)));
    save(checkpointFile, ...
        'RMSE_ORACLE', 'RMSE_ASOR', 'RMSE_GMM', ...
        'RMSE_STUDENT', 'RMSE_RCGP', 'RMSE_STD', ...
        'fitTime_ASOR', 'fitTime_GMM', ...
        'ASOR_outerConverged', 'GMM_outerConverged', ...
        'ASOR_hitMaxIter', 'GMM_hitMaxIter', ...
        'ASOR_iterMedian', 'GMM_iterMedian', ...
        'ASOR_iterMax', 'GMM_iterMax', ...
        'ASOR_finalRelMax', 'GMM_finalRelMax', ...
        'ASOR_innerConvFrac', 'GMM_innerConvFrac', ...
        'ASOR_innerHitCapFrac', 'GMM_innerHitCapFrac', ...
        'ASOR_lineSearchFailFrac', 'GMM_lineSearchFailFrac', ...
        'outlierEntryCount', 'outlierRowCount', ...
        'inputDimension', 'outputDimension', ...
        'datasetCfg', 'outlierCfg', 'caseName', ...
        'pOutList', 'pp', 'nTrain', 'nTest', 'nMC', ...
        'sigmaInit', 'fixedRcgprEpsilon', 'opts');
    fprintf('Checkpoint saved: %s\n', checkpointFile);
end

%% ========================================================================
% Package, plot, save, and summarize
% ========================================================================
results = struct();
results.experimentIndex = experimentIndex;
results.caseName = caseName;
results.dataset = datasetCfg.name;
results.datasetTag = datasetCfg.tag;
results.dataPath = datasetCfg.dataPath;
results.outputColumns = datasetCfg.outputColumns;
results.numOutputs = datasetCfg.numOutputs;
results.outlierDistribution = outlierCfg.label;
results.outlierModel = outlierCfg.model;
results.outlierScale = outlierCfg.scale;
results.outlierVariance = outlierVariance;
results.outlierMean = outlierMean;
results.outlierSecondMoment = outlierSecondMoment;
results.nTrain = nTrain;
results.nTest = nTest;
results.nMC = nMC;
results.pOutList = pOutList;
results.sigmaInit = sigmaInit;
results.oracleInlierVariance = oracleInlierVariance;
results.oracleOutlierVariance = oracleOutlierVariance;
results.methodNames = methodNames;
results.inputDimension = inputDimension;
results.outputDimension = outputDimension;
results.outlierEntryCount = outlierEntryCount;
results.outlierRowCount = outlierRowCount;

results.RMSE_ORACLE = RMSE_ORACLE;
results.RMSE_ASOR = RMSE_ASOR;
results.RMSE_GMM = RMSE_GMM;
results.RMSE_STUDENT = RMSE_STUDENT;
results.RMSE_RCGP = RMSE_RCGP;
results.RMSE_STD = RMSE_STD;

results.fitTime_ASOR = fitTime_ASOR;
results.fitTime_GMM = fitTime_GMM;
results.ASOR_outerConverged = ASOR_outerConverged;
results.GMM_outerConverged = GMM_outerConverged;
results.ASOR_hitMaxIter = ASOR_hitMaxIter;
results.GMM_hitMaxIter = GMM_hitMaxIter;
results.ASOR_iterMedian = ASOR_iterMedian;
results.GMM_iterMedian = GMM_iterMedian;
results.ASOR_iterMax = ASOR_iterMax;
results.GMM_iterMax = GMM_iterMax;
results.ASOR_finalRelMax = ASOR_finalRelMax;
results.GMM_finalRelMax = GMM_finalRelMax;
results.ASOR_innerConvFrac = ASOR_innerConvFrac;
results.GMM_innerConvFrac = GMM_innerConvFrac;
results.ASOR_innerHitCapFrac = ASOR_innerHitCapFrac;
results.GMM_innerHitCapFrac = GMM_innerHitCapFrac;
results.ASOR_lineSearchFailFrac = ASOR_lineSearchFailFrac;
results.GMM_lineSearchFailFrac = GMM_lineSearchFailFrac;

results.caseWallTimeSeconds = toc(caseWallTimer);
results.options = opts;

dataCell = {RMSE_ORACLE, RMSE_ASOR, RMSE_GMM, ...
            RMSE_STUDENT, RMSE_RCGP, RMSE_STD};

resultsFile = fullfile(caseOutputDir, [caseName '_results.mat']);
save(resultsFile, 'results', 'opts', 'methodNames', 'colors');

% Display and save this figure immediately before advancing to the next case.
figureBase = fullfile(caseOutputDir, [caseName '_pout_sweep_boxplot']);
plot_real_case_pout_sweep_boxplot( ...
    dataCell, pOutLabels, methodNames, colors, figureBase);
drawnow;

allResults(experimentIndex).results = results;
allResults(experimentIndex).resultsFile = resultsFile;
allResults(experimentIndex).figureBase = figureBase;

masterResultsFile = fullfile(outputRootDir, ...
    'RealData_Extensive_EightCases_NoOracleLeakage_results.mat');
save(masterResultsFile, 'allResults', 'datasets', 'outlierCases', ...
    'opts', 'methodNames', 'colors', 'nTrain', 'nTest', 'nMC', ...
    'pOutList', 'sigmaInit', 'fixedRcgprEpsilon');

fprintf('\n============================================================\n');
fprintf('FINAL SUMMARY: median and mean RMSE over MC trials\n');
fprintf('============================================================\n');

for pp = 1:nPout
    fprintf('\np_out = %.2f\n', pOutList(pp));
    for methodIdx = 1:numel(methodNames)
        fprintf('  %-15s median = %.4f | mean = %.4f\n', ...
            methodNames{methodIdx}, ...
            median(dataCell{methodIdx}(:, pp), 'omitnan'), ...
            mean(dataCell{methodIdx}(:, pp), 'omitnan'));
    end
end

fprintf('\nCase wall-clock time: %.2f minutes\n', ...
    results.caseWallTimeSeconds/60);
fprintf('Saved results: %s\n', resultsFile);
fprintf('Saved figure base: %s\n', figureBase);
fprintf('============================================================\n');

end
end

totalWallTimeSeconds = toc(totalWallTimer);
fprintf('\n\n============================================================\n');
fprintf('ALL %d REAL-DATA CASES COMPLETED\n', nExperiments);
fprintf('Total wall-clock time: %.2f hours\n', totalWallTimeSeconds/3600);
fprintf('Master results: %s\n', masterResultsFile);
fprintf('All figures/results: %s\n', outputRootDir);
fprintf('============================================================\n');

%% ========================================================================
% Local functions
% ========================================================================

function [outlierVariance, secondMoment, outlierMean] = ...
    outlier_stats_local(outlierModel, outlierScale)
%OUTLIER_STATS_LOCAL Statistics used only for reporting and Oracle setup.

    switch lower(outlierModel)
        case 'gaussian'
            % scale is standard deviation.
            outlierMean = 0;
            outlierVariance = outlierScale^2;
            secondMoment = outlierScale^2;

        case 'shifted_uniform'
            % scale = [a,b].
            a = outlierScale(1);
            b = outlierScale(2);
            if ~isfinite(a) || ~isfinite(b) || b <= a
                error('shifted_uniform requires finite endpoints [a,b] with b>a.');
            end
            outlierMean = 0.5 * (a + b);
            outlierVariance = (b - a)^2 / 12;
            secondMoment = (a^2 + a*b + b^2) / 3;

        otherwise
            error('Unsupported outlier model: %s', outlierModel);
    end
end


function outlierNoise = generate_outlier_noise_local( ...
    n, p, outlierModel, outlierScale)
%GENERATE_OUTLIER_NOISE_LOCAL Generate artificial training-target noise.
% This function is part of the experiment generator, not a fitted method.

    switch lower(outlierModel)
        case 'gaussian'
            outlierNoise = outlierScale * randn(n, p);

        case 'shifted_uniform'
            a = outlierScale(1);
            b = outlierScale(2);
            outlierNoise = a + (b - a) * rand(n, p);

        otherwise
            error('Unsupported outlier model: %s', outlierModel);
    end
end


function plot_real_case_pout_sweep_boxplot( ...
    dataCell, pOutLabels, methodNames, colors, figBase)
%PLOT_REAL_CASE_POUT_SWEEP_BOXPLOT
% Grouped filled boxplots over p_out, matching the simulation figure style.
%
% No title.
% Axis labels: RMSE and p_out.
% Automatically expands axes and exports png/eps/fig.

    nMethods = numel(dataCell);
    nPout = numel(pOutLabels);
    nMC = size(dataCell{1},1);

    fig = figure('Color','w', ...
        'Position',[80 80 1700 560], ...
        'WindowStyle','normal');

    hold on;

    centers = 1:nPout;
    delta = 0.10;
    offs = ((1:nMethods) - mean(1:nMethods)) * delta;
    bw = 0.09;

    h = gobjects(nMethods,1);
    allVals = [];

    for jj = 1:nMethods

        Xpos = repelem(centers + offs(jj), nMC).';
        Yvec = reshape(dataCell{jj}, [], 1);

        allVals = [allVals; Yvec]; %#ok<AGROW>

        h(jj) = boxchart(Xpos, Yvec, ...
            'BoxWidth', bw, ...
            'BoxFaceColor', colors(jj,:), ...
            'BoxFaceAlpha', 0.60, ...
            'MarkerColor', colors(jj,:), ...
            'LineWidth', 1.2, ...
            'DisplayName', methodNames{jj});
    end

    % Vertical separators between p_out groups.
    for ss = 1:(nPout-1)
        xline(ss + 0.5, ':', ...
            'Color', [0.45 0.45 0.45], ...
            'LineWidth', 1.4, ...
            'HandleVisibility', 'off');
    end

    xlim([0.5, nPout + 0.5]);

    finiteVals = allVals(isfinite(allVals));
    if isempty(finiteVals)
        yMax = 1;
    else
        yMax = 1.10 * max(finiteVals);
    end
    ylim([0, max(yMax, 0.1)]);

    xticks(centers);
    xticklabels(pOutLabels);

    ax = gca;

    % Axis formatting.
    ax.FontSize = 32;
    ax.LineWidth = 1.5;
    ax.TickLabelInterpreter = 'latex';
    ax.LabelFontSizeMultiplier = 1.0;
    ax.TitleFontSizeMultiplier = 1.0;

    hx = xlabel(ax, '$p_{\mathrm{out}}$', ...
        'Interpreter', 'latex');

    hy = ylabel(ax, '$\mathrm{RMSE}$', ...
        'Interpreter', 'latex');

    hx.FontSize = 35;
    hy.FontSize = 35;
    hx.FontWeight = 'normal';
    hy.FontWeight = 'normal';

    grid off;
    box on;

    % ------------------------------------------------------------
    % Expand axes to fill the figure better.
    % This affects both the displayed figure and saved files.
    % ------------------------------------------------------------
    ax.Units = 'normalized';
    ax.Position = [0.075 0.18 0.905 0.76];

    % Legend with visible boundary.
    lgd = legend(h, methodNames, ...
        'Location', 'northwest', ...
        'AutoUpdate', 'off', ...
        'Interpreter', 'none', ...
        'FontSize', 30, ...
        'Box', 'on');

    lgd.EdgeColor = [0 0 0];
    lgd.LineWidth = 1.2;
    lgd.Color = [1 1 1];

    % Make saved figure match on-screen layout.
    set(fig, 'PaperPositionMode', 'auto');
    set(fig, 'Renderer', 'painters');
    set(fig, 'InvertHardcopy', 'off');

    drawnow;

    % ------------------------------------------------------------
    % Export automatically.
    % ------------------------------------------------------------
    % ------------------------------------------------------------
    % Export automatically.
    % PNG and FIG are primary. EPS is attempted but will not stop the run.
    % ------------------------------------------------------------
    pngFile = [figBase '.png'];
    epsFile = [figBase '.eps'];
    figFile = [figBase '.fig'];
    
    try
        exportgraphics(fig, pngFile, 'Resolution', 300);
    catch ME
        warning('Could not save PNG file:\n  %s\nReason: %s', pngFile, ME.message);
    end
    
    try
        print(fig, epsFile, '-depsc', '-painters');
    catch ME
        warning('Could not save EPS file:\n  %s\nReason: %s', epsFile, ME.message);
    end
    
    try
        savefig(fig, figFile);
    catch ME
        warning('Could not save FIG file:\n  %s\nReason: %s', figFile, ME.message);
    end
    
    fprintf('\nSaved/attempted figure files:\n');
    fprintf('  %s\n', pngFile);
    fprintf('  %s\n', epsFile);
    fprintf('  %s\n', figFile);

    drawnow;
end


%% ========================================================================
% CORE ALGORITHMS AND SHARED HELPERS
% ========================================================================

function model = studentt_gpml_mogp_fit(X, Y, opts)
%STUDENTT_GPML_MOGP_FIT
% Fair independent-output Student-t GPR baseline using GPML.
%
% Fairness choices:
%   - no internal standardization of X
%   - no internal standardization of Y
%   - initialized from the same contaminated-data opts_run used by ASOR/GMM/RCGP
%   - likelihood sigma initialized from the shared opts.init_R
%   - kernel lengthscale initialized from opts.init_lengthscale
%   - signal std initialized from opts.init_signal_std
%   - mean initialized from the contaminated training output
%
% Each output dimension is fit by a separate scalar Student-t GP.

    [n, d] = size(X);
    p = size(Y,2);

    if isfield(opts, 'student_gpml_inference')
        inferName = upper(opts.student_gpml_inference);
    else
        inferName = 'VB';
    end

    if isfield(opts, 'student_gpml_nu')
        nuFixed = opts.student_gpml_nu;
    else
        nuFixed = 4;
    end

    if isfield(opts, 'student_gpml_optIters')
        optIters = opts.student_gpml_optIters;
    else
        optIters = -100;
    end

    % Same fixed R initialization given to all non-oracle methods.
    if ~isfield(opts, 'init_R')
        error('Student-t GP requires opts.init_R. Do not initialize R from data.');
    end
    
    Rinit = make_spd(opts.init_R, opts.jitter);

    model = struct();
    model.X = X;
    model.p = p;
    model.d = d;
    model.output = cell(p,1);
    model.inference = inferName;
    model.nu = nuFixed;

    for q = 1:p
        y = Y(:,q);

        meanfunc = {@meanConst};
        covfunc  = {@covSEard};
        likfunc  = {@likTnu_fixed_gpml};

        switch inferName
            case 'VB'
                inffunc = @infVB;
            case 'LA'
                inffunc = @infLaplace;
            otherwise
                error('Unknown opts.student_gpml_inference = %s. Use VB or LA.', inferName);
        end

        %% ------------------------------------------------------------
        % Fair contaminated-data initialization
        %% ------------------------------------------------------------

        % Mean initialized exactly from contaminated measurements.
        hyp0.mean = mean(y);

        % Kernel initialized from same opts_run used by other methods.
        ell0 = opts.init_lengthscale;
        if ~isfinite(ell0) || ell0 <= 0
            ell0 = 1.0;
        end

        sf0 = opts.init_signal_std;
        if ~isfinite(sf0) || sf0 <= 0
            sf0 = 1.0;
        end

        hyp0.cov = [log(ell0 * ones(d,1)); log(sf0)];

        % Student-t likelihood scale initialized from contaminated-data R.
        % This comes from the shared optsRun.init_R initialization.
        sigma0 = sqrt(max(Rinit(q,q), opts.jitter));

        if ~isfinite(sigma0) || sigma0 <= 0
            error('Invalid sigma0 from opts.init_R.');
        end

        hyp0.lik = log(max(sigma0, 1e-6));

        % Set fixed nu for likelihood wrapper.
        set_student_gpml_fixed_nu(nuFixed);

        try
            hyp = quiet_minimize_gpml(hyp0, @gp, optIters, ...
                inffunc, meanfunc, covfunc, likfunc, X, y);

            % Compute posterior once, silently.
            evalc('[post, nlZ] = feval(inffunc, hyp, meanfunc, covfunc, likfunc, X, y);');

            model.output{q}.ok = true;
            model.output{q}.hyp = hyp;
            model.output{q}.post = post;
            model.output{q}.nlZ = nlZ;
            model.output{q}.meanfunc = meanfunc;
            model.output{q}.covfunc = covfunc;
            model.output{q}.likfunc = likfunc;
            model.output{q}.inffunc = inffunc;
            model.output{q}.X = X;
            model.output{q}.y = y;

            % Useful diagnostics, not printed.
            model.output{q}.init_mean = hyp0.mean;
            model.output{q}.init_ell = ell0;
            model.output{q}.init_sf = sf0;
            model.output{q}.init_sigma = sigma0;
            model.output{q}.learned_sigma = exp(hyp.lik);
            model.output{q}.learned_ell = exp(hyp.cov(1:d));
            model.output{q}.learned_sf = exp(hyp.cov(end));

        catch ME
            warning('GPML Student-t failed for output %d: %s', q, ME.message);

            % Fallback also uses contaminated-data median.
            model.output{q}.ok = false;
            model.output{q}.fallback_value = mean(y);
        end
    end
end

function Ypred = studentt_gpml_mogp_predict(model, Xtest)
%STUDENTT_GPML_MOGP_PREDICT
% Predicts latent mean for each output dimension.
%
% No de-standardization is needed because RMSE is evaluated on the common
% standardized response scale.

    nTest = size(Xtest,1);
    p = model.p;

    Ypred = zeros(nTest,p);

    % Ensure likelihood wrapper has same fixed nu used in training.
    set_student_gpml_fixed_nu(model.nu);

    for q = 1:p
        out = model.output{q};

        if out.ok
            % Reuse posterior struct and suppress output.
            evalc('[~, ~, fmu, ~] = gp(out.hyp, out.inffunc, out.meanfunc, out.covfunc, out.likfunc, out.X, out.post, Xtest);');

            Ypred(:,q) = fmu;
        else
            Ypred(:,q) = out.fallback_value * ones(nTest,1);
        end
    end
end

function hyp_opt = quiet_minimize_gpml(hyp0, objfun, optIters, varargin)
%QUIET_MINIMIZE_GPML
% Suppresses GPML minimize printing:
%   Function evaluation 0; Value ...
%
% This only suppresses text output. It does not change optimization.

    evalc('hyp_opt = minimize(hyp0, objfun, optIters, varargin{:});');
end

function varargout = likTnu_fixed_gpml(hyp, y, mu, s2, inf, i)
%LIKTNU_FIXED_GPML
% Fixed-nu wrapper around GPML likT.
%
% GPML likT expects:
%   hyp_full = [log(nu - 1); log(sn)]
%
% This wrapper exposes only:
%   hyp = log(sn)
%
% The fixed nu is set by set_student_gpml_fixed_nu(nu).

    persistent nuFixed

    if isempty(nuFixed)
        nuFixed = 4;
    end

    if nargin == 1 && ischar(hyp) && strcmp(hyp, 'set_nu')
        nuFixed = y;
        varargout = {};
        return;
    end

    if nargin < 1
        varargout = {'1'};
        return;
    end

    if nargin < 3
        varargout = {'1'};
        return;
    end

    nuFixed = max(nuFixed, 1.01);
    hyp_full = [log(nuFixed - 1); hyp(:)];

    if nargin < 5
        [varargout{1:nargout}] = likT(hyp_full, y, mu, s2);
        return;
    end

    if nargin < 6
        [varargout{1:nargout}] = likT(hyp_full, y, mu, s2, inf);
        return;
    end

    % Wrapper hyp(1) corresponds to GPML likT hyp_full(2), log sigma.
    i_full = 2;
    [varargout{1:nargout}] = likT(hyp_full, y, mu, s2, inf, i_full);
end

function set_student_gpml_fixed_nu(nu)
%SET_STUDENT_GPML_FIXED_NU
% Sets the persistent nu used by likTnu_fixed_gpml.

    likTnu_fixed_gpml('set_nu', nu);
end

function model = standard_mogp_fit(X, Y, opts)
%STANDARD_MOGP_FIT Plain Gaussian multi-output GP baseline.
%
% This is the standard GP baseline:
%   y_i = f_i + e_i,   e_i ~ N(0, R0)
%
% R0 is the shared fixed initialization covariance from opts.init_R.
% Only the kernel parameters and constant mean are fitted.

    [n,d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    %% ------------------------------------------------------------
    % Fixed non-oracle measurement covariance
    %% ------------------------------------------------------------
    if ~isfield(opts,'init_R')
        error('Standard GP requires opts.init_R. Do not initialize R from data.');
    end

    R0 = make_spd(opts.init_R, opts.jitter);

    % Do not learn R for Standard GP.
    R_fixed = R0;

    %% ------------------------------------------------------------
    % Kernel initialization
    %% ------------------------------------------------------------
    logtheta0 = [log(opts.init_lengthscale*ones(d,1)); ...
                 log(opts.init_signal_std)];


    %% ------------------------------------------------------------
    % Optimize only kernel parameters: lengthscales and signal std
    %
    % Fairness change:
    %   - no multi-start fminsearch
    %   - same single-start backtracking gradient-descent style as ASOR/GMM
    %   - still optimizes the correct fixed-R Standard GP NLML
    %% ------------------------------------------------------------
    [logtheta, ~] = standard_fixedR_nlml_gradient_descent( ...
        logtheta0, X, Y, R_fixed, opts);
    
    bestVal = standard_mogp_fixedR_nlml(logtheta, X, Y, R_fixed, opts.jitter);

    %% ------------------------------------------------------------
    % Final exact Gaussian GP posterior with fixed R
    %% ------------------------------------------------------------
    Kx = ard_rbf_kernel(X, X, logtheta, opts.jitter);
    Ktheta = make_spd(kron(Kx, eye(p)), opts.jitter);

    Ky = make_spd(Ktheta + kron(eye(n), R_fixed), opts.jitter);

    [L,flag] = chol(Ky,'lower');
    if flag ~= 0
        Ky = make_spd(Ky, 1e-4);
        L = chol(Ky,'lower');
    end

    Lm = kron(ones(n,1), eye(p));

    KyInvY = L' \ (L \ yvec);
    KyInvM = L' \ (L \ Lm);

    A_m = Lm' * KyInvM;
    b_m = Lm' * KyInvY;

    m_hat = A_m \ b_m;
    mf = kron(ones(n,1), m_hat);

    alphaVec = L' \ (L \ (yvec - mf));

    mu_f = mf + Ktheta * alphaVec;
    Sigma_f = Ktheta - Ktheta * (L' \ (L \ Ktheta));
    Sigma_f = make_spd(Sigma_f, opts.jitter);

    model = struct();
    model.X = X;
    model.Y = Y;
    model.n = n;
    model.p = p;
    model.mu_f = mu_f;
    model.Sigma_f = Sigma_f;
    model.m_hat = m_hat;
    model.alpha_exact = alphaVec;
    model.logtheta = logtheta;
    model.Kinv = inv(make_spd(Ktheta, opts.jitter));
    model.R_fixed = R_fixed;
    model.nlml = bestVal;
end

function val = standard_mogp_fixedR_nlml(logtheta, X, Y, R_fixed, jitter)
%STANDARD_MOGP_FIXEDR_NLML
% Negative log marginal likelihood for Standard GP with fixed R.
%
% Optimizes only:
%   logtheta = [log ell_1, ..., log ell_d, log signal_std]
%
% Does not optimize sigma/R.

    [n,d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    %% ------------------------------------------------------------
    % Hard bounds to prevent pathological kernels
    %% ------------------------------------------------------------
    ell = exp(logtheta(1:d));
    sf  = exp(logtheta(end));

    Xrange = max(X,[],1) - min(X,[],1);
    Xrange(Xrange <= 0 | ~isfinite(Xrange)) = 1;

    if any(ell < 0.02*Xrange(:)) || any(ell > 5.0*Xrange(:)) || ...
       sf < 1e-4 || sf > 1e4
        val = 1e20;
        return;
    end

    R_fixed = make_spd(R_fixed, jitter);

    Kx = ard_rbf_kernel(X, X, logtheta, jitter);
    Ktheta = kron(Kx, eye(p));

    Ky = make_spd(Ktheta + kron(eye(n), R_fixed), jitter);

    [L,flag] = chol(Ky,'lower');

    if flag ~= 0
        val = 1e20;
        return;
    end

    %% ------------------------------------------------------------
    % GLS mean estimate under fixed R
    %% ------------------------------------------------------------
    Lm = kron(ones(n,1), eye(p));

    KyInvY = L' \ (L \ yvec);
    KyInvM = L' \ (L \ Lm);

    A_m = Lm' * KyInvM;
    b_m = Lm' * KyInvY;

    m_hat = A_m \ b_m;
    mf = kron(ones(n,1), m_hat);

    %% ------------------------------------------------------------
    % NLML
    %% ------------------------------------------------------------
    r = yvec - mf;
    alpha = L' \ (L \ r);

    val = 0.5*(r'*alpha) ...
        + sum(log(diag(L))) ...
        + 0.5*numel(yvec)*log(2*pi);

    if ~isfinite(val)
        val = 1e20;
    end
end

function Ypred = shared_mogp_predict(model, Xtest)
%SHARED_MOGP_PREDICT Predicts posterior means only.

    Xtrain = model.X;
    n = model.n;
    p = model.p;
    nTest = size(Xtest,1);
    Kinv = model.Kinv;
    mu_f = model.mu_f;
    m = model.m_hat;
    mf = kron(ones(n,1), m);
    logtheta = model.logtheta;
    ell = exp(logtheta(1:end-1));
    sf  = exp(logtheta(end));
    sf2 = sf^2;
    Ypred = zeros(nTest,p);

    for t = 1:nTest
        xstar = Xtest(t,:);
        kstar = ard_rbf_kernel_cross_single(xstar, Xtrain, ell, sf2);
        KstarX = kron(kstar, eye(p));

        if isfield(model, 'alpha_exact')
            mu_star = m + KstarX * model.alpha_exact;
        else
            mu_star = m + KstarX*Kinv*(mu_f - mf);
        end

        Ypred(t,:) = mu_star';
    end
end

function [J, grad] = kernel_variational_objective_grad(logtheta, X, mu_f, Sigma_f, m_hat, p, jitter)
%KERNEL_VARIATIONAL_OBJECTIVE_GRAD
% Computes ASOR variational kernel objective:
%
% J = 0.5*d'K^{-1}d + 0.5*tr(K^{-1}Sigma_f) + 0.5*log|K|
%
% and gradient with respect to logtheta:
% logtheta = [log ell_1, ..., log ell_d, log signal_std].
%
% This minimizes J. Therefore:
% dJ/dtheta = 0.5*tr((K^{-1} - K^{-1} C_f K^{-1}) dK/dtheta),
% where C_f = Sigma_f + d*d'.

    [n,d] = size(X);

    ell = exp(logtheta(1:d));
    sf2 = exp(logtheta(end))^2;

    Xscaled = X ./ ell(:)';
    D2 = squared_distance_matrix(Xscaled, Xscaled);

    % Base kernel without jitter, because jitter has zero derivative.
    Kbase = sf2 * exp(-0.5*D2);

    % Kernel used for objective.
    Kx = Kbase + jitter*eye(n);
    Ktheta = make_spd(kron(Kx, eye(p)), jitter);

    mf = kron(ones(n,1), m_hat);
    diff = mu_f - mf;

    [Rchol, flag] = chol(Ktheta, 'lower');
    if flag ~= 0
        J = 1e20;
        grad = zeros(numel(logtheta),1);
        return;
    end

    % K^{-1} times required matrices/vectors
    Kinv_diff = Rchol' \ (Rchol \ diff);
    Kinv_Sigma = Rchol' \ (Rchol \ Sigma_f);
    Kinv = Rchol' \ (Rchol \ eye(size(Ktheta)));

    logdetK = 2*sum(log(diag(Rchol)));

    J = 0.5*(diff' * Kinv_diff) ...
      + 0.5*trace(Kinv_Sigma) ...
      + 0.5*logdetK;

    if ~isfinite(J)
        J = 1e20;
        grad = zeros(numel(logtheta),1);
        return;
    end

    Cf = Sigma_f + diff*diff';

    % Gradient matrix for minimizing J:
    % dJ = 0.5 tr((K^{-1} - K^{-1}CfK^{-1}) dK)
    G = 0.5 * (Kinv - Kinv * Cf * Kinv);
    G = 0.5*(G + G');

    grad = zeros(d+1,1);

    % ARD lengthscale derivatives:
    % dK/dlog ell_q = Kbase .* ((x_q - x_q')^2 / ell_q^2)
    for q = 1:d
        xq = X(:,q);
        Dq2 = squared_distance_matrix(xq./ell(q), xq./ell(q));
        dKx = Kbase .* Dq2;
        dKtheta = kron(dKx, eye(p));
        grad(q) = sum(sum(G .* dKtheta));
    end

    % Signal std derivative:
    % K = sf^2 exp(...)
    % dK/dlog sf = 2Kbase
    dKx_sf = 2*Kbase;
    dKtheta_sf = kron(dKx_sf, eye(p));
    grad(end) = sum(sum(G .* dKtheta_sf));

    if any(~isfinite(grad))
        grad = zeros(size(grad));
    end
end

function [logtheta, info] = kernel_variational_gradient_descent(logtheta0, X, mu_f, Sigma_f, m_hat, p, opts)
%KERNEL_VARIATIONAL_GRADIENT_DESCENT
% Warm-started normalized gradient descent with Armijo backtracking.

    if ~strcmpi(opts.thetaOptimizer, 'gd_warmstart')
        error('This final script supports thetaOptimizer = gd_warmstart only.');
    end

    maxIter = opts.thetaGradMaxIter;
    step0 = opts.thetaGradStep;
    c1 = opts.thetaGradArmijo;
    minStep = opts.thetaGradMinStep;
    gradTol = opts.thetaGradTol;
    relTol = opts.thetaInnerRelTol;
    minInnerIter = opts.thetaMinInnerIter;
    stepIncrease = opts.thetaStepIncrease;
    minStart = opts.thetaGradStepMinStart;
    maxStart = opts.thetaGradStepMaxStart;

    logtheta = project_logtheta_bounds(logtheta0, X);
    [J, g] = kernel_variational_objective_grad( ...
        logtheta, X, mu_f, Sigma_f, m_hat, p, opts.jitter);

    info = struct();
    info.mode = 'gd_warmstart';
    info.J0 = J;
    info.Jfinal = J;
    info.gradNorm = norm(g);
    info.lastStep = 0;
    info.nSteps = 0;
    info.meanBacktracks = NaN;
    info.relJDrop = 0;
    info.lineSearchFailed = false;
    info.converged = false;
    info.convergedReason = 'none';
    info.hitMaxInner = false;
    info.finalRelJChange = inf;
    info.innerRelTol = relTol;
    info.gradTol = gradTol;

    if ~isfinite(J) || any(~isfinite(g))
        return;
    end

    stepStart = step0;
    backtrackTrace = nan(maxIter,1);
    stepTrace = nan(maxIter,1);
    Jstart = J;

    for kk = 1:maxIter

        gradNorm = norm(g);
        if gradNorm < gradTol
            info.converged = true;
            info.convergedReason = 'grad_norm';
            break;
        end

        direction = -g / max(1, gradNorm);
        trialStep = min(maxStart, max(minStart, stepStart));

        [accepted, candidate, Jcandidate, gcandidate, acceptedStep, nBacktrack] = ...
            theta_armijo_line_search_local( ...
            logtheta, J, g, direction, X, mu_f, Sigma_f, m_hat, p, ...
            opts.jitter, trialStep, c1, minStep);

        if ~accepted
            info.lineSearchFailed = true;
            break;
        end

        Jprevious = J;
        logtheta = candidate;
        J = Jcandidate;
        g = gcandidate;

        relativeChange = abs(Jprevious - J) / max(1, abs(Jprevious));

        info.nSteps = kk;
        info.lastStep = acceptedStep;
        info.Jfinal = J;
        info.gradNorm = norm(g);
        info.finalRelJChange = relativeChange;

        stepTrace(kk) = acceptedStep;
        backtrackTrace(kk) = nBacktrack;
        stepStart = min(maxStart, max(minStart, stepIncrease * acceptedStep));

        if kk >= minInnerIter && relativeChange < relTol
            info.converged = true;
            info.convergedReason = 'relative_objective';
            break;
        end
    end

    if info.nSteps >= maxIter && ~info.converged
        info.hitMaxInner = true;
    end

    validBacktracks = backtrackTrace(isfinite(backtrackTrace));
    validSteps = stepTrace(isfinite(stepTrace));

    if ~isempty(validBacktracks)
        info.meanBacktracks = mean(validBacktracks);
    end
    if ~isempty(validSteps)
        info.lastStep = validSteps(end);
    end

    info.Jfinal = J;
    info.gradNorm = norm(g);
    info.relJDrop = (Jstart - J) / max(1, abs(Jstart));
end
function [accepted, cand, Jcand, gcand, step, nBacktrack] = theta_armijo_line_search_local( ...
    logtheta, J, g, direction, X, mu_f, Sigma_f, m_hat, p, ...
    jitter, stepStart, c1, minStep)
%THETA_ARMIJO_LINE_SEARCH_LOCAL
% Local Armijo backtracking line search used by kernel_variational_gradient_descent.

    accepted = false;
    cand = logtheta;
    Jcand = J;
    gcand = g;
    step = stepStart;
    nBacktrack = 0;

    % Must be a descent direction.
    if ~all(isfinite(direction)) || g' * direction >= 0
        return;
    end

    while step >= minStep

        candTry = logtheta + step * direction;
        candTry = project_logtheta_bounds(candTry, X);

        [Jtry, gtry] = kernel_variational_objective_grad( ...
            candTry, X, mu_f, Sigma_f, m_hat, p, jitter);

        if isfinite(Jtry) && Jtry <= J + c1 * step * (g' * direction)
            accepted = true;
            cand = candTry;
            Jcand = Jtry;
            gcand = gtry;
            return;
        end

        step = 0.5 * step;
        nBacktrack = nBacktrack + 1;
    end
end

function logtheta = project_logtheta_bounds(logtheta, X)
%PROJECT_LOGTHETA_BOUNDS
% Keeps kernel parameters in a numerically reasonable range.

    d = size(X,2);

    ell = exp(logtheta(1:d));
    sf  = exp(logtheta(end));

    Xrange = max(X,[],1) - min(X,[],1);
    Xrange(Xrange <= 0 | ~isfinite(Xrange)) = 1;

    ell_min = 0.02 * Xrange(:);
    ell_max = 5.00 * Xrange(:);

    ell = min(max(ell(:), ell_min), ell_max);

    sf = min(max(sf, 1e-4), 1e4);

    logtheta = [log(ell); log(sf)];
end

function K = ard_rbf_kernel(X1, X2, logtheta, jitter)
    ell = exp(logtheta(1:end-1));
    sf2 = exp(logtheta(end))^2;
    D2 = squared_distance_matrix(X1 ./ ell(:)', X2 ./ ell(:)');
    K = sf2 * exp(-0.5*D2);
    if size(X1,1) == size(X2,1) && max(abs(X1(:)-X2(:))) < 1e-12
        K = K + jitter*eye(size(K,1));
    end
end

function kstar = ard_rbf_kernel_cross_single(xstar, Xtrain, ell, sf2)
    kstar = sf2 * exp(-0.5*sum((Xtrain ./ ell(:)' - xstar ./ ell(:)').^2, 2)');
end

function D2 = squared_distance_matrix(A, B)
    D2 = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'), 0);
end

%% ========================================================================
% PLOTTING AND UTILS
% ========================================================================

function yvec = stack_samples(Y)
    yvec = reshape(Y', [], 1);
end

function idx = sample_block(i,p)
    idx = (i-1)*p + (1:p);
end

function gamma = normalize_log_responsibilities(logResp)
    R = exp(logResp - max(logResp, [], 2));
    gamma = max(R ./ sum(R,2), 1e-12);
    gamma = gamma ./ sum(gamma,2);
end

function A = make_spd(A, jitter)
    A = (A + A')/2; jj = jitter;
    for k = 1:8
        [~, flag] = chol(A);
        if flag == 0, return; end
        A = A + jj*eye(size(A,1)); jj = 10*jj;
    end
    A = A + 1e-3*eye(size(A,1));
end

function Omega = logistic_inverse_from_log_ratio(logRatio)
    Omega = min(max(1.0 ./ (1.0 + exp(min(max(logRatio, -50), 50))), 1e-12), 1-1e-12);
end

function d = median_pairwise_distance(X)

    X = double(X);
    n = size(X,1);

    if n <= 1
        d = 1.0;
        return;
    end

    max_n = min(n, 500);
    idx = randperm(n, max_n);
    Xs = X(idx,:);

    D = sqrt(squared_distance_matrix(Xs, Xs));
    vals = D(triu(true(size(D)),1));

    vals = vals(isfinite(vals) & vals > 0);

    if isempty(vals)
        d = 1.0;
    else
        d = median(vals, 'omitnan');
    end
end

function model = independent_standard_gpr_fit(X, Y, opts)
%INDEPENDENT_STANDARD_GPR_FIT
% Fits one scalar-output Standard GPR model per output dimension.
%
% Fairness:
%   - one scalar GP per output dimension
%   - one ARD kernel per output dimension
%   - one fixed scalar noise variance per output dimension
%
% Therefore Standard GPR no longer uses a stacked multi-output GP with a
% shared kernel across all outputs.

    [n, d] = size(X);
    p = size(Y,2);

    model = struct();
    model.X = X;
    model.n = n;
    model.d = d;
    model.p = p;
    model.output = cell(p,1);
    model.method = 'Independent-Standard-GPR';

    if ~isfield(opts, 'init_R')
        error('independent_standard_gpr_fit requires opts.init_R.');
    end

    for q = 1:p

        opts_q = opts;

        % Each scalar Standard GP receives the corresponding scalar
        % diagonal entry of the same shared initialization covariance.
        opts_q.init_R = opts.init_R(q,q);

        % Fit scalar Standard GPR to output q.
        model.output{q} = standard_mogp_fit(X, Y(:,q), opts_q);
    end
end

function Ypred = independent_standard_gpr_predict(model, Xtest)
%INDEPENDENT_STANDARD_GPR_PREDICT
% Predicts all output dimensions from independent scalar Standard GPR models.

    nTest = size(Xtest,1);
    p = model.p;

    Ypred = zeros(nTest,p);

    for q = 1:p
        yq_pred = shared_mogp_predict(model.output{q}, Xtest);
        Ypred(:,q) = yq_pred(:,1);
    end
end

function model = independent_oracle_gmm_style_gpr_fit(X, Y, isOutlierEntry, opts)
%INDEPENDENT_ORACLE_GMM_STYLE_GPR_FIT
% Perfect-detection GMM-style Oracle.
%
% This Oracle does NOT train on clean data.
% It trains on contaminated Y, but it knows:
%   1. the exact entry-wise inlier/outlier labels,
%   2. the true inlier variance,
%   3. the true total outlier residual variance.
%
% It is structurally matched to independent scalar-output GMM-GPR:
% one scalar GP per output dimension.

    [n, d] = size(X);
    p = size(Y,2);

    if ~isequal(size(isOutlierEntry), size(Y))
        error('isOutlierEntry must have the same size as Y.');
    end

    if ~isfield(opts, 'oracle_inlier_var')
        error('Oracle requires opts.oracle_inlier_var.');
    end

    if ~isfield(opts, 'oracle_outlier_var')
        error('Oracle requires opts.oracle_outlier_var.');
    end

    model = struct();
    model.X = X;
    model.n = n;
    model.d = d;
    model.p = p;
    model.output = cell(p,1);
    model.method = 'Oracle';

    for q = 1:p

        isOut_q = isOutlierEntry(:,q);

        model.output{q} = oracle_scalar_known_labels_gpr_fit( ...
            X, Y(:,q), isOut_q, opts);
    end
end

function Ypred = independent_oracle_gmm_style_gpr_predict(model, Xtest)
%INDEPENDENT_ORACLE_GMM_STYLE_GPR_PREDICT
% Predicts all output dimensions from independent scalar Oracle models.

    nTest = size(Xtest,1);
    p = model.p;

    Ypred = zeros(nTest,p);

    for q = 1:p
        yq_pred = shared_mogp_predict(model.output{q}, Xtest);
        Ypred(:,q) = yq_pred(:,1);
    end
end

function model = oracle_scalar_known_labels_gpr_fit(X, y, isOutlier, opts)
%ORACLE_SCALAR_KNOWN_LABELS_GPR_FIT
% Scalar-output Oracle GP with known heteroscedastic component variance.
%
% For each scalar observation:
%
%   if inlier:
%       noise variance = opts.oracle_inlier_var
%
%   if outlier:
%       noise variance = opts.oracle_outlier_var
%
% This is the perfect-label, perfect-variance analogue of a two-component
% GMM-GPR model.

    X = double(X);
    y = double(y(:));
    isOutlier = logical(isOutlier(:));

    [n,d] = size(X);

    if numel(y) ~= n || numel(isOutlier) ~= n
        error('X, y, and isOutlier have inconsistent sizes.');
    end

    oracle_inlier_var  = opts.oracle_inlier_var;
    oracle_outlier_var = opts.oracle_outlier_var;

    if oracle_inlier_var <= 0 || oracle_outlier_var <= 0
        error('Oracle variances must be positive.');
    end

    noiseVarVec = oracle_inlier_var * ones(n,1);
    noiseVarVec(isOutlier) = oracle_outlier_var;

    %% ------------------------------------------------------------
    % Kernel initialization
    %% ------------------------------------------------------------
    ell0 = opts.init_lengthscale;

    if ~isfinite(ell0) || ell0 <= 0
        ell0 = 1.0;
    end

    sf0 = opts.init_signal_std;

    if ~isfinite(sf0) || sf0 <= 0
        sf0 = 1.0;
    end

    logtheta0 = [log(ell0 * ones(d,1)); log(sf0)];

    %% ------------------------------------------------------------
    % Optimize kernel hyperparameters with known heteroscedastic variances
    %% ------------------------------------------------------------
    [logtheta, oracleInfo] = oracle_hetero_fixed_noise_nlml_gradient_descent( ...
        logtheta0, X, y, noiseVarVec, opts);

    %% ------------------------------------------------------------
    % Final exact GP posterior with known heteroscedastic observation noise
    %% ------------------------------------------------------------
    Ktheta = ard_rbf_kernel(X, X, logtheta, opts.jitter);
    Ktheta = make_spd(Ktheta, opts.jitter);

    Ky = make_spd(Ktheta + diag(noiseVarVec), opts.jitter);

    [L,flag] = chol(Ky, 'lower');

    if flag ~= 0
        Ky = make_spd(Ky, 1e-4);
        L = chol(Ky, 'lower');
    end

    % GLS estimate of constant mean under known heteroscedastic noise.
    Lm = ones(n,1);

    KyInvY = L' \ (L \ y);
    KyInvM = L' \ (L \ Lm);

    m_hat = (Lm' * KyInvM) \ (Lm' * KyInvY);

    mf = m_hat * ones(n,1);

    alphaVec = L' \ (L \ (y - mf));

    mu_f = mf + Ktheta * alphaVec;

    Sigma_f = Ktheta - Ktheta * (L' \ (L \ Ktheta));
    Sigma_f = make_spd(Sigma_f, opts.jitter);

    model = struct();
    model.X = X;
    model.Y = y;
    model.n = n;
    model.p = 1;
    model.mu_f = mu_f;
    model.Sigma_f = Sigma_f;
    model.m_hat = m_hat;
    model.alpha_exact = alphaVec;
    model.logtheta = logtheta;
    model.Kinv = inv(make_spd(Ktheta, opts.jitter));

    model.noiseVarVec = noiseVarVec;
    model.isOutlierKnown = isOutlier;

    model.oracle_inlier_var = oracle_inlier_var;
    model.oracle_outlier_var = oracle_outlier_var;
    model.oracleInfo = oracleInfo;
    model.method = 'Oracle';
end

function [logtheta, info] = standard_fixedR_nlml_gradient_descent(logtheta0, X, Y, R_fixed, opts)
%STANDARD_FIXEDR_NLML_GRADIENT_DESCENT
% Single-start numerical-gradient descent for Standard GP fixed-R NLML.
%
% This keeps Standard GP fair:
%   - it still optimizes the proper exact Gaussian GP marginal likelihood
%   - it does not learn R
%   - it does not use multi-start fminsearch
%   - it uses the same style of projected backtracking gradient descent as ASOR/GMM

    if isfield(opts, 'fixedGPGradMaxIter')
        maxIter = opts.fixedGPGradMaxIter;
    elseif isfield(opts, 'thetaGradMaxIter')
        maxIter = opts.thetaGradMaxIter;
    else
        maxIter = opts.thetaMaxIter;
    end

    if isfield(opts, 'thetaGradStep')
        step0 = opts.thetaGradStep;
    else
        step0 = 0.10;
    end

    if isfield(opts, 'thetaGradArmijo')
        c1 = opts.thetaGradArmijo;
    else
        c1 = 1e-4;
    end

    if isfield(opts, 'thetaGradMinStep')
        minStep = opts.thetaGradMinStep;
    else
        minStep = 1e-7;
    end

    logtheta = project_logtheta_bounds(logtheta0, X);

    J = standard_mogp_fixedR_nlml(logtheta, X, Y, R_fixed, opts.jitter);
    g = numerical_grad_standard_fixedR(logtheta, X, Y, R_fixed, opts.jitter);

    info = struct();
    info.J0 = J;
    info.Jfinal = J;
    info.gradNorm = norm(g);
    info.lastStep = 0;
    info.nSteps = 0;

    if ~isfinite(J) || any(~isfinite(g))
        return;
    end

    for kk = 1:maxIter

        gradNorm = norm(g);

        if gradNorm < 1e-8
            break;
        end

        % Same normalized descent direction idea used in your ASOR/GMM update.
        direction = -g / max(1, gradNorm);

        step = step0;
        accepted = false;

        while step >= minStep

            cand = logtheta + step * direction;
            cand = project_logtheta_bounds(cand, X);

            Jcand = standard_mogp_fixedR_nlml(cand, X, Y, R_fixed, opts.jitter);

            if isfinite(Jcand) && Jcand <= J + c1 * step * (g' * direction)
                logtheta = cand;
                J = Jcand;
                g = numerical_grad_standard_fixedR(logtheta, X, Y, R_fixed, opts.jitter);
                accepted = true;
                break;
            end

            step = 0.5 * step;
        end

        if ~accepted
            break;
        end

        info.nSteps = kk;
        info.lastStep = step;
        info.Jfinal = J;
        info.gradNorm = norm(g);
    end
end

function [logtheta, info] = oracle_hetero_fixed_noise_nlml_gradient_descent( ...
    logtheta0, X, y, noiseVarVec, opts)
%ORACLE_HETERO_FIXED_NOISE_NLML_GRADIENT_DESCENT
% Single-start projected numerical-gradient descent for Oracle GP with
% known heteroscedastic observation variances.

    if isfield(opts, 'fixedGPGradMaxIter')
        maxIter = opts.fixedGPGradMaxIter;
    elseif isfield(opts, 'thetaGradMaxIter')
        maxIter = opts.thetaGradMaxIter;
    else
        maxIter = opts.thetaMaxIter;
    end

    if isfield(opts, 'thetaGradStep')
        step0 = opts.thetaGradStep;
    else
        step0 = 0.10;
    end

    if isfield(opts, 'thetaGradArmijo')
        c1 = opts.thetaGradArmijo;
    else
        c1 = 1e-4;
    end

    if isfield(opts, 'thetaGradMinStep')
        minStep = opts.thetaGradMinStep;
    else
        minStep = 1e-7;
    end

    logtheta = project_logtheta_bounds(logtheta0, X);

    J = oracle_scalar_hetero_nlml(logtheta, X, y, noiseVarVec, opts.jitter);
    g = numerical_grad_oracle_hetero(logtheta, X, y, noiseVarVec, opts.jitter);

    info = struct();
    info.J0 = J;
    info.Jfinal = J;
    info.gradNorm = norm(g);
    info.lastStep = 0;
    info.nSteps = 0;

    if ~isfinite(J) || any(~isfinite(g))
        return;
    end

    for kk = 1:maxIter

        gradNorm = norm(g);

        if gradNorm < 1e-8
            break;
        end

        direction = -g / max(1, gradNorm);

        step = step0;
        accepted = false;

        while step >= minStep

            cand = logtheta + step * direction;
            cand = project_logtheta_bounds(cand, X);

            Jcand = oracle_scalar_hetero_nlml( ...
                cand, X, y, noiseVarVec, opts.jitter);

            if isfinite(Jcand) && Jcand <= J + c1 * step * (g' * direction)
                logtheta = cand;
                J = Jcand;
                g = numerical_grad_oracle_hetero( ...
                    logtheta, X, y, noiseVarVec, opts.jitter);

                accepted = true;
                break;
            end

            step = 0.5 * step;
        end

        if ~accepted
            break;
        end

        info.nSteps = kk;
        info.lastStep = step;
        info.Jfinal = J;
        info.gradNorm = norm(g);
    end
end

function val = oracle_scalar_hetero_nlml(logtheta, X, y, noiseVarVec, jitter)
%ORACLE_SCALAR_HETERO_NLML
% Negative log marginal likelihood for scalar GP with known heteroscedastic
% observation variances.

    X = double(X);
    y = double(y(:));
    noiseVarVec = double(noiseVarVec(:));

    [n,d] = size(X);

    if numel(y) ~= n || numel(noiseVarVec) ~= n
        error('Inconsistent sizes in oracle_scalar_hetero_nlml.');
    end

    ell = exp(logtheta(1:d));
    sf  = exp(logtheta(end));

    Xrange = max(X,[],1) - min(X,[],1);
    Xrange(Xrange <= 0 | ~isfinite(Xrange)) = 1;

    if any(ell < 0.02*Xrange(:)) || any(ell > 5.0*Xrange(:)) || ...
       sf < 1e-4 || sf > 1e4
        val = 1e20;
        return;
    end

    if any(noiseVarVec <= 0) || any(~isfinite(noiseVarVec))
        val = 1e20;
        return;
    end

    Ktheta = ard_rbf_kernel(X, X, logtheta, jitter);
    Ky = make_spd(Ktheta + diag(noiseVarVec), jitter);

    [L,flag] = chol(Ky, 'lower');

    if flag ~= 0
        val = 1e20;
        return;
    end

    % GLS estimate of constant mean.
    Lm = ones(n,1);

    KyInvY = L' \ (L \ y);
    KyInvM = L' \ (L \ Lm);

    A_m = Lm' * KyInvM;
    b_m = Lm' * KyInvY;

    if abs(A_m) < realmin || ~isfinite(A_m)
        val = 1e20;
        return;
    end

    m_hat = A_m \ b_m;

    r = y - m_hat;

    alpha = L' \ (L \ r);

    val = 0.5*(r' * alpha) ...
        + sum(log(diag(L))) ...
        + 0.5*n*log(2*pi);

    if ~isfinite(val)
        val = 1e20;
    end
end

function g = numerical_grad_oracle_hetero(logtheta, X, y, noiseVarVec, jitter)
%NUMERICAL_GRAD_ORACLE_HETERO
% Central finite-difference gradient for Oracle heteroscedastic GP NLML.

    g = zeros(size(logtheta));

    f0 = oracle_scalar_hetero_nlml(logtheta, X, y, noiseVarVec, jitter);

    if ~isfinite(f0)
        return;
    end

    for k = 1:numel(logtheta)

        h = 1e-4 * max(1, abs(logtheta(k)));

        zp = logtheta;
        zm = logtheta;

        zp(k) = zp(k) + h;
        zm(k) = zm(k) - h;

        fp = oracle_scalar_hetero_nlml(zp, X, y, noiseVarVec, jitter);
        fm = oracle_scalar_hetero_nlml(zm, X, y, noiseVarVec, jitter);

        if isfinite(fp) && isfinite(fm)
            g(k) = (fp - fm) / (2*h);
        else
            g(k) = 0;
        end
    end

    if any(~isfinite(g))
        g = zeros(size(logtheta));
    end
end

function g = numerical_grad_standard_fixedR(logtheta, X, Y, R_fixed, jitter)
%NUMERICAL_GRAD_STANDARD_FIXEDR
% Central finite-difference gradient of Standard GP fixed-R NLML.

    g = zeros(size(logtheta));

    f0 = standard_mogp_fixedR_nlml(logtheta, X, Y, R_fixed, jitter);

    if ~isfinite(f0)
        return;
    end

    for k = 1:numel(logtheta)

        h = 1e-4 * max(1, abs(logtheta(k)));

        zp = logtheta;
        zm = logtheta;

        zp(k) = zp(k) + h;
        zm(k) = zm(k) - h;

        fp = standard_mogp_fixedR_nlml(zp, X, Y, R_fixed, jitter);
        fm = standard_mogp_fixedR_nlml(zm, X, Y, R_fixed, jitter);

        if isfinite(fp) && isfinite(fm)
            g(k) = (fp - fm) / (2*h);
        else
            g(k) = 0;
        end
    end

    if any(~isfinite(g))
        g = zeros(size(logtheta));
    end
end

function model = independent_rcgp_gpr_fit(X, Y, opts)
%INDEPENDENT_RCGP_GPR_FIT
% Faithful paper-style RCGPR baseline.
%
% The original RCGP paper is scalar-output. Therefore, for multi-output
% experiments, we fit one scalar RCGP independently per output dimension.
%
% Method name in figures/tables remains RCGPR.

    [n, d] = size(X);
    p = size(Y,2);

    model = struct();
    model.X = X;
    model.n = n;
    model.d = d;
    model.p = p;
    model.output = cell(p,1);
    model.method = 'RCGPR';

    if ~isfield(opts, 'init_R')
        error('independent_rcgp_gpr_fit requires opts.init_R.');
    end

    for q = 1:p

        rcgpOpts_q = struct();

        % Paper RCGP hyperparameters/settings
        rcgpOpts_q.epsilon = opts.rcgp_epsilon;
        rcgpOpts_q.useShrinkageTerm = opts.rcgp_useShrinkageTerm;
        rcgpOpts_q.shrinkageConvention = opts.rcgp_shrinkageConvention;

        % Fair initialization from the same contaminated-data setup.
        % The paper estimates sigma, but we initialize it using the same
        % initial variance given to the other non-oracle methods.
        rcgpOpts_q.initLengthscale = opts.init_lengthscale;
        rcgpOpts_q.initSignalStd   = opts.init_signal_std;
        rcgpOpts_q.initNoiseStd    = sqrt(max(opts.init_R(q,q), opts.jitter));

        % Numerical settings
        rcgpOpts_q.jitter = opts.jitter;
        rcgpOpts_q.maxIter = opts.rcgp_maxIter;
        rcgpOpts_q.lbfgsMemory = opts.rcgp_lbfgsMemory;
        rcgpOpts_q.gradTol = opts.rcgp_gradTol;
        rcgpOpts_q.stepTol = opts.rcgp_stepTol;
        rcgpOpts_q.armijo = opts.rcgp_armijo;
        rcgpOpts_q.minStep = opts.rcgp_minStep;
        rcgpOpts_q.verbose = opts.rcgp_verbose;

        model.output{q} = rcgp_scalar_faithful_fit( ...
            X, Y(:,q), rcgpOpts_q);
    end
end

function Ypred = independent_rcgp_gpr_predict(model, Xtest)
%INDEPENDENT_RCGP_GPR_PREDICT
% Predicts all output dimensions from independent faithful scalar RCGP models.

    nTest = size(Xtest,1);
    p = model.p;

    Ypred = zeros(nTest,p);

    for q = 1:p
        mu_q = rcgp_scalar_faithful_predict(model.output{q}, Xtest);

        Ypred(:,q) = mu_q;
    end
end

function model = rcgp_scalar_faithful_fit(X, y, opts)
%RCGP_SCALAR_FAITHFUL_FIT
% Faithful scalar-output RCGP implementation.
%
% Paper-style structure:
%   m = mean(y)
%   c = Q_n(1 - epsilon) of |y_i - m|
%   J_i = 1 + ((y_i-m)/c)^2
%   z_i = d_i - 2*sigma^2*d_i/(c^2+d_i^2)
%   A = K + sigma^2*diag(J_i)
%
% Hyperparameters ell, signal std, and sigma are estimated by minimizing
% the negative analytical LOO predictive log score.

    X = double(X);
    y = double(y(:));

    [n,d] = size(X);

    if numel(y) ~= n
        error('X and y have inconsistent numbers of samples.');
    end

    opts = rcgp_scalar_fill_defaults(opts, X, y);

    % Paper uses constant prior mean equal to sample mean.
    meanConstant = mean(y);

    dvec0 = y - meanConstant;
    c = rcgp_empirical_quantile_local(abs(dvec0), 1 - opts.epsilon);
    c = max(c, 1e-8);

    initEll = opts.initLengthscale;

    if isscalar(initEll)
        initEll = initEll * ones(d,1);
    else
        initEll = initEll(:);
    end

    logHyp0 = [log(initEll); ...
               log(opts.initSignalStd); ...
               log(opts.initNoiseStd)];

    xRange = max(X,[],1) - min(X,[],1);
    xRange(~isfinite(xRange) | xRange <= 0) = 1;

    yScale = std(y);

    if ~isfinite(yScale) || yScale < 1e-8
        yScale = 1;
    end

    lower = [log(max(1e-3*xRange(:), 1e-5)); ...
             log(max(1e-5*yScale, 1e-6)); ...
             log(max(1e-5*yScale, 1e-6))];

    upper = [log(max(100*xRange(:), 1)); ...
             log(max(100*yScale, 10)); ...
             log(max(20*yScale, 5))];

    logHyp0 = min(max(logHyp0, lower), upper);

    objective = @(logHyp) rcgp_scalar_loo_objective_grad( ...
        logHyp, X, y, meanConstant, c, opts);

    [logHyp, fitInfo] = rcgp_lbfgs_box( ...
        objective, logHyp0, lower, upper, opts);

    ell = exp(logHyp(1:d));
    sf = exp(logHyp(d+1));
    sigma = exp(logHyp(d+2));
    sigma2 = sigma^2;

    Kbase = rcgp_se_ard_kernel(X, X, ell, sf);
    K = Kbase + opts.jitter*eye(n);

    dvec = y - meanConstant;

    Jdiag = 1 + (dvec/c).^2;
    robustNoiseDiag = sigma2 * Jdiag;

    if opts.useShrinkageTerm
        shrinkageSign = rcgp_scalar_shrinkage_sign(opts);
        z = dvec ...
            + shrinkageSign*2*sigma2*dvec ./ ...
              (c^2 + dvec.^2);
    else
        z = dvec;
    end

    A = K + diag(robustNoiseDiag);

    L = rcgp_stable_chol(A, opts.jitter);

    alpha = L' \ (L \ z);

    beta = sigma / sqrt(2);
    normalizedWeight = 1 ./ sqrt(Jdiag);
    actualWeight = beta * normalizedWeight;

    model = struct();
    model.X = X;
    model.y = y;
    model.n = n;
    model.d = d;

    model.meanConstant = meanConstant;
    model.epsilon = opts.epsilon;
    model.c = c;

    model.lengthscale = ell;
    model.signalStd = sf;
    model.signalVar = sf^2;
    model.noiseStd = sigma;
    model.noiseVar = sigma2;

    model.beta = beta;
    model.Jdiag = Jdiag;
    model.robustNoiseDiag = robustNoiseDiag;
    model.normalizedWeight = normalizedWeight;
    model.actualWeight = actualWeight;
    model.z = z;

    model.K = K;
    model.L = L;
    model.alpha = alpha;

    model.fitInfo = fitInfo;
    model.opts = opts;
    model.method = 'RCGPR';
end

function muStar = rcgp_scalar_faithful_predict(model, Xstar)
%RCGP_SCALAR_FAITHFUL_PREDICT
% Closed-form paper-style RCGP posterior-mean prediction.

    Xstar = double(Xstar);

    KtrainStar = rcgp_se_ard_kernel( ...
        model.X, Xstar, model.lengthscale, model.signalStd);

    muStar = model.meanConstant + KtrainStar' * model.alpha;
end

function [nll, grad] = rcgp_scalar_loo_objective_grad( ...
    logHyp, X, y, m, c, opts)
%RCGP_SCALAR_LOO_OBJECTIVE_GRAD
% Negative analytical RCGP LOO predictive log score and gradient.

    X = double(X);
    y = double(y(:));

    [n,d] = size(X);

    ell = exp(logHyp(1:d));
    sf = exp(logHyp(d+1));
    sigma = exp(logHyp(d+2));
    sigma2 = sigma^2;

    Kbase = rcgp_se_ard_kernel(X, X, ell, sf);
    K = Kbase + opts.jitter*eye(n);

    dvec = y - m;

    denominatorShrinkage = c^2 + dvec.^2;

    Jdiag = 1 + (dvec/c).^2;
    robustNoiseDiag = sigma2 * Jdiag;

    if opts.useShrinkageTerm
        shrinkageSign = rcgp_scalar_shrinkage_sign(opts);
        z = dvec + shrinkageSign*2*sigma2*dvec ./ denominatorShrinkage;
    else
        shrinkageSign = 0;
        z = dvec;
    end

    A = K + diag(robustNoiseDiag);

    try
        L = rcgp_stable_chol(A, opts.jitter);
    catch
        nll = realmax/100;
        grad = zeros(size(logHyp));
        return;
    end

    Q = L' \ (L \ eye(n));
    Q = 0.5*(Q + Q');

    qdiag = diag(Q);

    if any(~isfinite(qdiag)) || any(qdiag <= 0)
        nll = realmax/100;
        grad = zeros(size(logHyp));
        return;
    end

    alpha = Q*z;
    invQdiag = 1 ./ qdiag;

    looMean = m + z - alpha .* invQdiag;

    looLatentVar = invQdiag - robustNoiseDiag;
    looObservedVar = looLatentVar + sigma2;

    if any(~isfinite(looObservedVar)) || any(looObservedVar <= 1e-12)
        nll = realmax/100;
        grad = zeros(size(logHyp));
        return;
    end

    predictionError = y - looMean;

    nll = 0.5 * sum( ...
        log(2*pi*looObservedVar) ...
        + predictionError.^2 ./ looObservedVar);

    if ~isfinite(nll)
        nll = realmax/100;
        grad = zeros(size(logHyp));
        return;
    end

    nParams = d + 2;
    grad = zeros(nParams,1);

    for paramIdx = 1:nParams

        dK = zeros(n,n);
        dNoiseDiag = zeros(n,1);
        dSigma2 = 0;
        dz = zeros(n,1);

        if paramIdx <= d

            xq = X(:,paramIdx);
            Dq2 = ((xq - xq')./ell(paramIdx)).^2;
            dK = Kbase .* Dq2;

        elseif paramIdx == d + 1

            dK = 2*Kbase;

        else

            dSigma2 = 2*sigma2;
            dNoiseDiag = 2*robustNoiseDiag;

            if opts.useShrinkageTerm
                dz = shrinkageSign*4*sigma2*dvec ./ denominatorShrinkage;
            end
        end

        dA = dK + diag(dNoiseDiag);

        QdA = Q*dA;
        dqdiag = -sum(QdA .* Q', 2);

        dInvQdiag = -dqdiag ./ (qdiag.^2);

        dAlpha = -Q*(dA*alpha) + Q*dz;

        dLooMean = dz ...
            - dAlpha .* invQdiag ...
            + alpha .* dqdiag ./ (qdiag.^2);

        dLooObservedVar = dInvQdiag - dNoiseDiag + dSigma2;

        grad(paramIdx) = sum( ...
            0.5 * (1./looObservedVar ...
                   - predictionError.^2 ./ looObservedVar.^2) ...
                  .* dLooObservedVar ...
            - (predictionError ./ looObservedVar) .* dLooMean);
    end

    if any(~isfinite(grad))
        grad = zeros(size(logHyp));
    end
end

function [x, info] = rcgp_lbfgs_box(fun, x0, lower, upper, opts)
%RCGP_LBFGS_BOX
% Small self-contained projected L-BFGS optimizer.

    x = min(max(x0(:), lower(:)), upper(:));

    [f, g] = fun(x);

    S = cell(0,1);
    Y = cell(0,1);
    RHO = zeros(0,1);

    converged = false;
    lastStep = 0;
    iter = 0;

    if opts.verbose
        fprintf('\nRCGP L-BFGS hyperparameter optimization\n');
        fprintf(' iter | objective       | ||grad||_inf | step\n');
        fprintf('------------------------------------------------\n');
        fprintf('%5d | %15.7e | %12.4e | %8.2e\n', ...
            0, f, norm(g,inf), 0);
    end

    for iter = 1:opts.maxIter

        if norm(g,inf) <= opts.gradTol
            converged = true;
            break;
        end

        q = g;
        nMemory = numel(S);
        alphaMemory = zeros(nMemory,1);

        for ii = nMemory:-1:1
            alphaMemory(ii) = RHO(ii) * (S{ii}'*q);
            q = q - alphaMemory(ii)*Y{ii};
        end

        if nMemory > 0
            sy = S{end}'*Y{end};
            yy = Y{end}'*Y{end};
            gamma = sy / max(yy, realmin);
        else
            gamma = 1;
        end

        r = gamma*q;

        for ii = 1:nMemory
            betaMemory = RHO(ii) * (Y{ii}'*r);
            r = r + S{ii}*(alphaMemory(ii) - betaMemory);
        end

        direction = -r;

        if ~all(isfinite(direction)) || g'*direction >= 0
            direction = -g;
        end

        step = 1.0;
        accepted = false;

        while step >= opts.minStep

            candidate = min(max(x + step*direction, lower), upper);
            displacement = candidate - x;

            if norm(displacement,inf) <= opts.stepTol
                step = 0.5*step;
                continue;
            end

            [fCandidate, gCandidate] = fun(candidate);

            if isfinite(fCandidate) && ...
                    fCandidate <= f + opts.armijo*(g'*displacement)
                accepted = true;
                break;
            end

            step = 0.5*step;
        end

        if ~accepted
            break;
        end

        s = candidate - x;
        yChange = gCandidate - g;
        curvature = s'*yChange;

        if curvature > 1e-12*max(1, norm(s)*norm(yChange))

            if numel(S) == opts.lbfgsMemory
                S(1) = [];
                Y(1) = [];
                RHO(1) = [];
            end

            S{end+1,1} = s;
            Y{end+1,1} = yChange;
            RHO(end+1,1) = 1/curvature;
        end

        x = candidate;
        f = fCandidate;
        g = gCandidate;
        lastStep = step;

        if opts.verbose
            fprintf('%5d | %15.7e | %12.4e | %8.2e\n', ...
                iter, f, norm(g,inf), step);
        end

        if norm(s,inf) <= opts.stepTol
            converged = true;
            break;
        end
    end

    info = struct();
    info.finalObjective = f;
    info.finalGradient = g;
    info.finalGradientInfNorm = norm(g,inf);
    info.iterations = iter;
    info.converged = converged;
    info.lastStep = lastStep;
end

function opts = rcgp_scalar_fill_defaults(opts, X, y)

    if nargin < 1 || isempty(opts)
        opts = struct();
    end

    d = size(X,2);

    if ~isfield(opts,'epsilon'), opts.epsilon = 0.05; end
    if ~isfield(opts,'initLengthscale'), opts.initLengthscale = ones(d,1); end
    if ~isfield(opts,'initSignalStd'), opts.initSignalStd = 1.0; end
    if ~isfield(opts,'initNoiseStd'), opts.initNoiseStd = 1.0; end
    if ~isfield(opts,'jitter'), opts.jitter = 1e-8; end
    if ~isfield(opts,'maxIter'), opts.maxIter = 100; end
    if ~isfield(opts,'lbfgsMemory'), opts.lbfgsMemory = 10; end
    if ~isfield(opts,'gradTol'), opts.gradTol = 1e-5; end
    if ~isfield(opts,'stepTol'), opts.stepTol = 1e-10; end
    if ~isfield(opts,'armijo'), opts.armijo = 1e-4; end
    if ~isfield(opts,'minStep'), opts.minStep = 1e-10; end
    if ~isfield(opts,'verbose'), opts.verbose = false; end
    if ~isfield(opts,'useShrinkageTerm'), opts.useShrinkageTerm = true; end
    if ~isfield(opts,'shrinkageConvention'), opts.shrinkageConvention = 'section3'; end

    if opts.epsilon < 0 || opts.epsilon > 1
        error('opts.epsilon must lie in [0,1].');
    end

    if opts.initSignalStd <= 0 || opts.initNoiseStd <= 0
        error('Initial standard deviations must be positive.');
    end

    if any(opts.initLengthscale(:) <= 0)
        error('Initial lengthscales must be positive.');
    end

    if any(~isfinite(y)) || any(~isfinite(X(:)))
        error('X and y must contain only finite values.');
    end
end

function signValue = rcgp_scalar_shrinkage_sign(opts)
%RCGP_SCALAR_SHRINKAGE_SIGN
%
% 'section3':
%   z = d - 2*sigma^2*d/(c^2+d^2)
%
% 'proposition_literal':
%   z = d + 2*sigma^2*d/(c^2+d^2)

    switch lower(opts.shrinkageConvention)

        case 'section3'
            signValue = -1;

        case 'proposition_literal'
            signValue = +1;

        otherwise
            error('Unknown shrinkageConvention: %s', opts.shrinkageConvention);
    end
end

function K = rcgp_se_ard_kernel(X1, X2, ell, signalStd)

    X1 = double(X1);
    X2 = double(X2);

    ell = ell(:)';

    if isscalar(ell)
        ell = ell * ones(1,size(X1,2));
    end

    if size(X1,2) ~= size(X2,2) || numel(ell) ~= size(X1,2)
        error('Kernel input dimensions are inconsistent.');
    end

    X1scaled = X1 ./ ell;
    X2scaled = X2 ./ ell;

    D2 = max( ...
        sum(X1scaled.^2,2) ...
        + sum(X2scaled.^2,2)' ...
        - 2*(X1scaled*X2scaled'), ...
        0);

    K = signalStd^2 * exp(-0.5*D2);
end

function L = rcgp_stable_chol(A, initialJitter)

    A = 0.5*(A + A');
    n = size(A,1);

    extraJitter = 0;

    for attempt = 1:10

        [L,flag] = chol(A + extraJitter*eye(n), 'lower');

        if flag == 0
            return;
        end

        if extraJitter == 0
            extraJitter = initialJitter;
        else
            extraJitter = 10*extraJitter;
        end
    end

    error('RCGP Cholesky factorization failed after repeated jitter increases.');
end

function q = rcgp_empirical_quantile_local(values, probability)

    values = sort(values(:));
    n = numel(values);

    probability = min(max(probability,0),1);

    if n == 0
        error('Cannot compute a quantile of an empty vector.');

    elseif n == 1
        q = values(1);
        return;
    end

    position = 1 + (n-1)*probability;
    lowerIdx = floor(position);
    upperIdx = ceil(position);
    fraction = position - lowerIdx;

    q = (1-fraction)*values(lowerIdx) + fraction*values(upperIdx);
end

%% ========================================================================
% UPDATED ASOR/GMM LR-DIAGNOSTIC LOCAL FUNCTIONS
% ========================================================================

function criterion = get_outer_criterion_local(opts)
    if isfield(opts, 'outerConvergenceCriterion')
        criterion = lower(char(opts.outerConvergenceCriterion));
    else
        criterion = 'obj_single';
    end
end

function tf = outer_convergence_met_local(relTrace, stateRelTrace, it, opts)
    tf = false;
    if it <= opts.thetaBurnIn
        return;
    end
    tol = opts.tol;
    criterion = get_outer_criterion_local(opts);
    objNow = relTrace(it);
    stateNow = stateRelTrace(it);
    switch criterion
        case 'obj_single'
            tf = isfinite(objNow) && objNow < tol;
        case 'obj_window3_max'
            tf = window_max_below_tol_local(relTrace, it, 3, tol);
        case 'obj_window5_max'
            tf = window_max_below_tol_local(relTrace, it, 5, tol);
        case 'state_single'
            tf = isfinite(stateNow) && stateNow < tol;
        case 'state_window3'
            tf = window_max_below_tol_local(stateRelTrace, it, 3, tol);
        case 'hybrid_single'
            tf = isfinite(objNow) && objNow < tol && isfinite(stateNow) && stateNow < tol;
        case 'hybrid_window3'
            tf = window_max_below_tol_local(relTrace, it, 3, tol) && ...
                 window_max_below_tol_local(stateRelTrace, it, 3, tol);
        otherwise
            error('Unknown outerConvergenceCriterion = %s', criterion);
    end
end

function tf = window_max_below_tol_local(traceVec, it, win, tol)
    tf = false;
    if it < win
        return;
    end
    vals = traceVec((it-win+1):it);
    vals = vals(:);
    tf = all(isfinite(vals)) && max(vals) < tol;
end

function [relChange, newPrevStateVec] = outer_state_rel_change_local(stateVec, prevStateVec)
    stateVec = double(stateVec(:));
    if isempty(prevStateVec) || numel(prevStateVec) ~= numel(stateVec)
        relChange = inf;
    else
        relChange = norm(stateVec - prevStateVec) / max(1, norm(prevStateVec));
    end
    if ~isfinite(relChange), relChange = inf; end
    newPrevStateVec = stateVec;
end

function diag = collect_independent_fit_diagnostics_cap(model)
%COLLECT_INDEPENDENT_FIT_DIAGNOSTICS_CAP
% Aggregates scalar-output diagnostics and records both outer and inner
% convergence diagnostics.

    p = model.p;

    nIter = nan(p,1);
    conv = false(p,1);
    hitMax = false(p,1);
    finalRel = nan(p,1);
    thetaAttempted = zeros(p,1);
    thetaAccepted = zeros(p,1);
    thetaStepMed = nan(p,1);
    thetaGradMed = nan(p,1);
    thetaInnerStepsMed = nan(p,1);
    thetaInnerStepsTotal = nan(p,1);
    thetaInnerConvFracByOutput = nan(p,1);
    thetaInnerHitCapFracByOutput = nan(p,1);
    thetaLineSearchFailFracByOutput = nan(p,1);
    thetaInnerFinalRelMedByOutput = nan(p,1);

    for q = 1:p
        out = model.output{q};

        if isfield(out, 'fitInfo')
            fi = out.fitInfo;

            if isfield(fi, 'nIter'), nIter(q) = fi.nIter; end
            if isfield(fi, 'converged'), conv(q) = fi.converged; end
            if isfield(fi, 'hitMaxIter'), hitMax(q) = fi.hitMaxIter; end
            if isfield(fi, 'finalRelChange'), finalRel(q) = fi.finalRelChange; end
            if isfield(fi, 'thetaUpdateCount'), thetaAttempted(q) = fi.thetaUpdateCount; end
            if isfield(fi, 'thetaAcceptedCount'), thetaAccepted(q) = fi.thetaAcceptedCount; end

            if isfield(fi, 'thetaStepTrace')
                st = fi.thetaStepTrace(:);
                st = st(isfinite(st) & st > 0);
                if ~isempty(st), thetaStepMed(q) = median(st, 'omitnan'); end
            end

            if isfield(fi, 'thetaGradNormTrace')
                gt = fi.thetaGradNormTrace(:);
                gt = gt(isfinite(gt));
                if ~isempty(gt), thetaGradMed(q) = median(gt, 'omitnan'); end
            end

            if isfield(fi, 'thetaNStepsTrace')
                ns = fi.thetaNStepsTrace(:);
                ns = ns(isfinite(ns) & ns >= 0);
                if ~isempty(ns)
                    thetaInnerStepsMed(q) = median(ns, 'omitnan');
                    thetaInnerStepsTotal(q) = sum(ns);
                end
            end

            if isfield(fi, 'thetaUpdateCount') && fi.thetaUpdateCount > 0
                if isfield(fi, 'thetaInnerConvergedCount')
                    thetaInnerConvFracByOutput(q) = fi.thetaInnerConvergedCount / fi.thetaUpdateCount;
                end
                if isfield(fi, 'thetaInnerHitMaxCount')
                    thetaInnerHitCapFracByOutput(q) = fi.thetaInnerHitMaxCount / fi.thetaUpdateCount;
                end
                if isfield(fi, 'thetaLineSearchFailureCount')
                    thetaLineSearchFailFracByOutput(q) = fi.thetaLineSearchFailureCount / fi.thetaUpdateCount;
                end
            end

            if isfield(fi, 'thetaInnerFinalRelTrace')
                rt = fi.thetaInnerFinalRelTrace(:);
                rt = rt(isfinite(rt) & rt >= 0);
                if ~isempty(rt), thetaInnerFinalRelMedByOutput(q) = median(rt, 'omitnan'); end
            end
        end
    end

    diag = struct();
    diag.nIter = nIter;
    diag.converged = conv;
    diag.hitMaxIter = hitMax;
    diag.finalRelChange = finalRel;
    diag.thetaAttempted = thetaAttempted;
    diag.thetaAccepted = thetaAccepted;
    diag.thetaStepMedianByOutput = thetaStepMed;
    diag.thetaGradMedianByOutput = thetaGradMed;
    diag.thetaInnerStepsMedianByOutput = thetaInnerStepsMed;
    diag.thetaInnerStepsTotalByOutput = thetaInnerStepsTotal;
    diag.thetaInnerConvFracByOutput = thetaInnerConvFracByOutput;
    diag.thetaInnerHitCapFracByOutput = thetaInnerHitCapFracByOutput;
    diag.thetaLineSearchFailFracByOutput = thetaLineSearchFailFracByOutput;
    diag.thetaInnerFinalRelMedByOutput = thetaInnerFinalRelMedByOutput;

    diag.allConverged = all(conv);
    diag.anyHitMaxIter = any(hitMax);
    diag.iterMedian = median(nIter, 'omitnan');
    diag.iterMax = max(nIter, [], 'omitnan');
    diag.finalRelMax = max(finalRel, [], 'omitnan');
    diag.finalRelMedian = median(finalRel, 'omitnan');

    totalAttempted = sum(thetaAttempted);
    totalAccepted = sum(thetaAccepted);

    if totalAttempted > 0
        diag.thetaAcceptFrac = totalAccepted / totalAttempted;
    else
        diag.thetaAcceptFrac = NaN;
    end

    diag.thetaStepMedian = median(thetaStepMed, 'omitnan');
    diag.thetaGradMedian = median(thetaGradMed, 'omitnan');
    diag.thetaInnerStepsMedian = median(thetaInnerStepsMed, 'omitnan');
    diag.thetaInnerStepsTotalMedian = median(thetaInnerStepsTotal, 'omitnan');
    diag.thetaInnerStepsTotalMax = max(thetaInnerStepsTotal, [], 'omitnan');
    diag.thetaInnerConvFrac = median(thetaInnerConvFracByOutput, 'omitnan');
    diag.thetaInnerHitCapFrac = median(thetaInnerHitCapFracByOutput, 'omitnan');
    diag.thetaLineSearchFailFrac = median(thetaLineSearchFailFracByOutput, 'omitnan');
    diag.thetaInnerFinalRelMedian = median(thetaInnerFinalRelMedByOutput, 'omitnan');
end

function model = independent_asor_gpr_fit_lrdiag(X, Y, opts)
%INDEPENDENT_ASOR_GPR_FIT_LRDIAG
% Independent scalar-output ASOR-GPR with fit diagnostics.

    [n, d] = size(X);
    p = size(Y,2);

    model = struct();
    model.X = X;
    model.n = n;
    model.d = d;
    model.p = p;
    model.output = cell(p,1);
    model.method = 'Independent-ASOR-GPR-LRDiag';

    if ~isfield(opts, 'init_R')
        error('independent_asor_gpr_fit_lrdiag requires opts.init_R.');
    end

    for q = 1:p
        opts_q = opts;
        opts_q.init_R = opts.init_R(q,q);

        if isfield(opts_q, 'nu0_base_offset')
            opts_q.nu0 = 1 + opts_q.nu0_base_offset;
        else
            opts_q.nu0 = 3;
        end

        model.output{q} = asor_mogp_fit_grad_lrdiag(X, Y(:,q), opts_q);
    end
end

function Ypred = independent_asor_gpr_predict_lrdiag(model, Xtest)
%INDEPENDENT_ASOR_GPR_PREDICT_LRDIAG

    nTest = size(Xtest,1);
    p = model.p;
    Ypred = zeros(nTest,p);

    for q = 1:p
        yq_pred = shared_mogp_predict(model.output{q}, Xtest);
        Ypred(:,q) = yq_pred(:,1);
    end
end

function model = independent_gmm_gpr_fit_lrdiag(X, Y, opts)
%INDEPENDENT_GMM_GPR_FIT_LRDIAG
% Independent scalar-output GMM-GPR with fit diagnostics.

    [n, d] = size(X);
    p = size(Y,2);

    model = struct();
    model.X = X;
    model.n = n;
    model.d = d;
    model.p = p;
    model.output = cell(p,1);
    model.method = 'Independent-GMM-GPR-LRDiag';

    if ~isfield(opts, 'init_R')
        error('independent_gmm_gpr_fit_lrdiag requires opts.init_R.');
    end

    for q = 1:p
        opts_q = opts;
        opts_q.init_R = opts.init_R(q,q);
        model.output{q} = gmm_mogp_fit_lrdiag(X, Y(:,q), opts_q);
    end
end

function Ypred = independent_gmm_gpr_predict_lrdiag(model, Xtest)
%INDEPENDENT_GMM_GPR_PREDICT_LRDIAG

    nTest = size(Xtest,1);
    p = model.p;
    Ypred = zeros(nTest,p);

    for q = 1:p
        yq_pred = shared_mogp_predict(model.output{q}, Xtest);
        Ypred(:,q) = yq_pred(:,1);
    end
end

function model = asor_mogp_fit_grad_lrdiag(X, Y, opts)
%ASOR_MOGP_FIT_GRAD_LRDIAG
% Scalar ASOR-GPR with convergence diagnostics.

    [n, d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    m_hat = median(Y,1)';

    if ~isfield(opts, 'init_R')
        error('ASOR-GPR-LRDiag requires opts.init_R.');
    end

    Sigma_hat = make_spd(opts.init_R, opts.jitter);

    b_hat = opts.init_b;
    S0 = opts.S0_scale * eye(p);

    w = ones(n,1);
    Omega = ones(n,1);

    logtheta = [log(opts.init_lengthscale*ones(d,1)); log(opts.init_signal_std)];

    Kx = ard_rbf_kernel(X, X, logtheta, opts.jitter);
    Ktheta = kron(Kx, eye(p));

    mu_f = kron(ones(n,1), m_hat);
    Sigma_f = eye(n*p);

    prevObj = inf;

    objTrace = nan(opts.maxIter,1);
    relTrace = nan(opts.maxIter,1);
    sigmaTrace = nan(opts.maxIter,1);
    thetaGradNormTrace = nan(opts.maxIter,1);
    thetaStepTrace = nan(opts.maxIter,1);
    thetaNStepsTrace = nan(opts.maxIter,1);
    thetaUpdateCount = 0;
    thetaAcceptedCount = 0;
    thetaFallbackCount = 0;
    thetaInnerConvergedCount = 0;
    thetaInnerHitMaxCount = 0;
    thetaLineSearchFailureCount = 0;
    thetaInnerFinalRelTrace = nan(opts.maxIter,1);
    finalRelChange = inf;
    stateRelTrace = nan(opts.maxIter,1);
    finalStateRelChange = inf;
    prevOuterStateVec = [];
    convergedFlag = false;

    for it = 1:opts.maxIter

        Sigma_hat = make_spd(Sigma_hat, opts.jitter);
        SigmaInv = inv(Sigma_hat);

        Lambda_w = kron(diag(w), SigmaInv);

        Ktheta = make_spd(Ktheta, opts.jitter);
        Kinv = inv(Ktheta);

        Precision_f = make_spd(Kinv + Lambda_w, opts.jitter);
        Sigma_f = inv(Precision_f);

        mf = kron(ones(n,1), m_hat);
        mu_f = Sigma_f * (Kinv*mf + Lambda_w*yvec);

        alpha_shape = opts.a + p/2;

        R = zeros(n,1);
        beta = zeros(n,1);
        S_blocks = cell(n,1);

        for i = 1:n
            idx = sample_block(i,p);

            mu_i = mu_f(idx);
            V_i  = Sigma_f(idx,idx);
            y_i  = Y(i,:)';

            diff_i = y_i - mu_i;

            R(i) = diff_i' * SigmaInv * diff_i + trace(SigmaInv * V_i);
            beta(i) = b_hat + 0.5*R(i);

            logRatio = log((1-opts.theta0)/opts.theta0) ...
                     + gammaln(alpha_shape) ...
                     - gammaln(opts.a) ...
                     + opts.a*log(max(b_hat, realmin)) ...
                     - alpha_shape*log(max(beta(i), realmin)) ...
                     + 0.5*R(i);

            Omega(i) = logistic_inverse_from_log_ratio(logRatio);

            w(i) = Omega(i) + (1 - Omega(i)) * alpha_shape / beta(i);
            w(i) = max(w(i), opts.minWeight);

            S_blocks{i} = diff_i*diff_i' + V_i;
        end

        S = zeros(p,p);
        for i = 1:n
            S = S + w(i)*S_blocks{i};
        end

        % Original IW-mode ASOR covariance update used in the manuscript.
        denom_R = n + opts.nu0 + p + 1;
        Sigma_hat = (S + S0) / max(denom_R, realmin);

        Sigma_hat = make_spd(Sigma_hat, opts.jitter);

        numerator = opts.A - 1 + opts.a * sum(1 - Omega);
        denominator = opts.B + sum((1 - Omega) .* (alpha_shape ./ beta));
        b_hat = max(numerator / max(denominator, realmin), realmin);

        Lm = kron(ones(n,1), eye(p));
        m_hat = (Lm' * Kinv * Lm) \ (Lm' * Kinv * mu_f);

        if opts.learnTheta && it > opts.thetaBurnIn && mod(it, opts.thetaUpdateEvery) == 0

            thetaUpdateCount = thetaUpdateCount + 1;
            logtheta_old = logtheta;

            [logtheta_new, thetaInfo] = kernel_variational_gradient_descent( ...
                logtheta, X, mu_f, Sigma_f, m_hat, p, opts);

            thetaGradNormTrace(it) = thetaInfo.gradNorm;
            thetaStepTrace(it) = thetaInfo.lastStep;
            thetaNStepsTrace(it) = thetaInfo.nSteps;
            if isfield(thetaInfo, 'finalRelJChange')
                thetaInnerFinalRelTrace(it) = thetaInfo.finalRelJChange;
            end
            if isfield(thetaInfo, 'converged') && thetaInfo.converged
                thetaInnerConvergedCount = thetaInnerConvergedCount + 1;
            end
            if isfield(thetaInfo, 'hitMaxInner') && thetaInfo.hitMaxInner
                thetaInnerHitMaxCount = thetaInnerHitMaxCount + 1;
            end
            if isfield(thetaInfo, 'lineSearchFailed') && thetaInfo.lineSearchFailed
                thetaLineSearchFailureCount = thetaLineSearchFailureCount + 1;
            end

            if any(~isfinite(logtheta_new))
                thetaFallbackCount = thetaFallbackCount + 1;
                logtheta = logtheta_old;
            else
                logtheta = logtheta_new;
            end

            if isfield(thetaInfo, 'nSteps') && thetaInfo.nSteps > 0
                thetaAcceptedCount = thetaAcceptedCount + 1;
            end

            Kx = ard_rbf_kernel(X, X, logtheta, opts.jitter);
            Ktheta = kron(Kx, eye(p));
        end

        weightedResidualObj = sum(w .* R);
        objTrace(it) = weightedResidualObj;
        sigmaTrace(it) = sqrt(Sigma_hat(1,1));

        relChange = abs(prevObj - weightedResidualObj) / max(1, abs(prevObj));
        relTrace(it) = relChange;
        finalRelChange = relChange;

        currentOuterStateVec = [m_hat(:); logtheta(:); Sigma_hat(:); b_hat; w(:); Omega(:)];
        [stateRel, prevOuterStateVec] = outer_state_rel_change_local( ...
            currentOuterStateVec, prevOuterStateVec);
        stateRelTrace(it) = stateRel;
        finalStateRelChange = stateRel;

        if outer_convergence_met_local(relTrace, stateRelTrace, it, opts)
            convergedFlag = true;
            break;
        end

        prevObj = weightedResidualObj;
    end

    Ktheta_final = make_spd(kron(ard_rbf_kernel(X, X, logtheta, opts.jitter), eye(p)), opts.jitter);

    fitInfo = struct();
    fitInfo.nIter = it;
    fitInfo.maxIter = opts.maxIter;
    fitInfo.converged = convergedFlag;
    fitInfo.hitMaxIter = it >= opts.maxIter && ~convergedFlag;
    fitInfo.finalRelChange = finalRelChange;
    fitInfo.finalStateRelChange = finalStateRelChange;
    fitInfo.outerConvergenceCriterion = get_outer_criterion_local(opts);
    fitInfo.objTrace = objTrace(1:it);
    fitInfo.stateRelTrace = stateRelTrace(1:it);
    fitInfo.relTrace = relTrace(1:it);
    fitInfo.sigmaTrace = sigmaTrace(1:it);
    fitInfo.thetaGradNormTrace = thetaGradNormTrace(1:it);
    fitInfo.thetaStepTrace = thetaStepTrace(1:it);
    fitInfo.thetaNStepsTrace = thetaNStepsTrace(1:it);
    fitInfo.thetaUpdateCount = thetaUpdateCount;
    fitInfo.thetaAcceptedCount = thetaAcceptedCount;
    fitInfo.thetaFallbackCount = thetaFallbackCount;
    fitInfo.thetaInnerConvergedCount = thetaInnerConvergedCount;
    fitInfo.thetaInnerHitMaxCount = thetaInnerHitMaxCount;
    fitInfo.thetaLineSearchFailureCount = thetaLineSearchFailureCount;
    fitInfo.thetaInnerFinalRelTrace = thetaInnerFinalRelTrace(1:it);
    fitInfo.thetaGradStep = opts.thetaGradStep;
    fitInfo.thetaGradMaxIter = opts.thetaGradMaxIter;

    model = struct('X',X, 'Y',Y, 'n',n, 'p',p, ...
        'mu_f',mu_f, ...
        'Sigma_f',Sigma_f, ...
        'm_hat',m_hat, ...
        'logtheta',logtheta, ...
        'Kinv',inv(Ktheta_final), ...
        'w',w, ...
        'Omega',Omega, ...
        'Sigma_hat',Sigma_hat, ...
        'b_hat',b_hat, ...
        'method','ASOR-GPR-LRDiag', ...
        'fitInfo', fitInfo);
end

function model = gmm_mogp_fit_lrdiag(X, Y, opts)
%GMM_MOGP_FIT_LRDIAG
% Scalar two-component GMM-GPR with convergence diagnostics.

    [n, d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    m_hat = median(Y,1)';

    if ~isfield(opts, 'init_R')
        error('GMM-GPR-LRDiag requires opts.init_R.');
    end
    
    if ~isfield(opts, 'gmmOutlierInitMultiplier') || ...
            ~isfield(opts, 'gmmNominalInitMultiplier')
        error('GMM initialization multipliers must be supplied in opts.');
    end

    Sigma_base = make_spd(opts.init_R, opts.jitter);
    Sigma_out0 = make_spd( ...
        opts.gmmOutlierInitMultiplier * Sigma_base, opts.jitter);
    Sigma_nom0 = make_spd( ...
        opts.gmmNominalInitMultiplier * Sigma_base, opts.jitter);
    
    Sigma_comp = {Sigma_out0; Sigma_nom0};
    alpha = [0.5, 0.5];

    logtheta = [log(opts.init_lengthscale*ones(d,1)); log(opts.init_signal_std)];

    mu_f = kron(ones(n,1), m_hat);
    Sigma_f = eye(n*p);
    gamma = repmat(alpha, n, 1);

    prevObj = inf;

    objTrace = nan(opts.maxIter,1);
    relTrace = nan(opts.maxIter,1);
    thetaGradNormTrace = nan(opts.maxIter,1);
    thetaStepTrace = nan(opts.maxIter,1);
    thetaNStepsTrace = nan(opts.maxIter,1);
    thetaUpdateCount = 0;
    thetaAcceptedCount = 0;
    thetaFallbackCount = 0;
    thetaInnerConvergedCount = 0;
    thetaInnerHitMaxCount = 0;
    thetaLineSearchFailureCount = 0;
    thetaInnerFinalRelTrace = nan(opts.maxIter,1);
    finalRelChange = inf;
    stateRelTrace = nan(opts.maxIter,1);
    finalStateRelChange = inf;
    prevOuterStateVec = [];
    convergedFlag = false;

    for it = 1:opts.maxIter

        Lambda = zeros(n*p, n*p);

        invSigma = cell(2,1);
        logDetSigma = zeros(2,1);

        for j = 1:2
            Sj = make_spd(Sigma_comp{j}, opts.jitter);
            invSigma{j} = inv(Sj);
            logDetSigma(j) = log(det(Sj));
        end

        for i = 1:n
            idx = sample_block(i,p);
            Lambda_i = gamma(i,1)*invSigma{1} + gamma(i,2)*invSigma{2};
            Lambda(idx,idx) = make_spd(Lambda_i, opts.jitter);
        end

        Kx = ard_rbf_kernel(X, X, logtheta, opts.jitter);
        Ktheta = make_spd(kron(Kx, eye(p)), opts.jitter);
        Kinv = inv(Ktheta);

        Precision_f = make_spd(Kinv + Lambda, opts.jitter);
        Sigma_f = inv(Precision_f);

        mf = kron(ones(n,1), m_hat);
        mu_f = Sigma_f * (Kinv*mf + Lambda*yvec);

        logResp = zeros(n,2);

        for i = 1:n
            idx = sample_block(i,p);
            diff_i = Y(i,:)' - mu_f(idx);
            V_i = Sigma_f(idx,idx);

            for j = 1:2
                expected_quad = diff_i' * invSigma{j} * diff_i ...
                              + trace(invSigma{j} * V_i);

                logResp(i,j) = log(max(alpha(j), realmin)) ...
                    - 0.5*p*log(2*pi) ...
                    - 0.5*logDetSigma(j) ...
                    - 0.5*expected_quad;
            end
        end

        gamma = normalize_log_responsibilities(logResp);

        alpha = max(mean(gamma,1), 1e-8);
        alpha = alpha / sum(alpha);

        for j = 1:2
            S_j = zeros(p,p);
            sum_gamma = sum(gamma(:,j));

            for i = 1:n
                idx = sample_block(i,p);
                diff_i = Y(i,:)' - mu_f(idx);
                V_i = Sigma_f(idx,idx);

                S_j = S_j + gamma(i,j) * (diff_i*diff_i' + V_i);
            end

            Sigma_comp{j} = make_spd( ...
                S_j / max(sum_gamma, realmin), opts.jitter);
        end

        if trace(Sigma_comp{2}) > trace(Sigma_comp{1})
            tmpS = Sigma_comp{1};
            Sigma_comp{1} = Sigma_comp{2};
            Sigma_comp{2} = tmpS;

            alpha = fliplr(alpha);
            gamma = fliplr(gamma);
        end

        Lm = kron(ones(n,1), eye(p));
        m_hat = (Lm' * Kinv * Lm) \ (Lm' * Kinv * mu_f);

        if opts.learnTheta && it > opts.thetaBurnIn && mod(it, opts.thetaUpdateEvery) == 0

            thetaUpdateCount = thetaUpdateCount + 1;
            logtheta_old = logtheta;

            [logtheta_new, thetaInfo] = kernel_variational_gradient_descent( ...
                logtheta, X, mu_f, Sigma_f, m_hat, p, opts);

            thetaGradNormTrace(it) = thetaInfo.gradNorm;
            thetaStepTrace(it) = thetaInfo.lastStep;
            thetaNStepsTrace(it) = thetaInfo.nSteps;
            if isfield(thetaInfo, 'finalRelJChange')
                thetaInnerFinalRelTrace(it) = thetaInfo.finalRelJChange;
            end
            if isfield(thetaInfo, 'converged') && thetaInfo.converged
                thetaInnerConvergedCount = thetaInnerConvergedCount + 1;
            end
            if isfield(thetaInfo, 'hitMaxInner') && thetaInfo.hitMaxInner
                thetaInnerHitMaxCount = thetaInnerHitMaxCount + 1;
            end
            if isfield(thetaInfo, 'lineSearchFailed') && thetaInfo.lineSearchFailed
                thetaLineSearchFailureCount = thetaLineSearchFailureCount + 1;
            end

            if any(~isfinite(logtheta_new))
                thetaFallbackCount = thetaFallbackCount + 1;
                logtheta = logtheta_old;
            else
                logtheta = logtheta_new;
            end

            if isfield(thetaInfo, 'nSteps') && thetaInfo.nSteps > 0
                thetaAcceptedCount = thetaAcceptedCount + 1;
            end
        end

        obj = 0;
        for i = 1:n
            idx = sample_block(i,p);
            diff_i = Y(i,:)' - mu_f(idx);
            V_i = Sigma_f(idx,idx);

            for j = 1:2
                Sj = make_spd(Sigma_comp{j}, opts.jitter);
                SjInv = inv(Sj);

                obj = obj + gamma(i,j) * ...
                    (diff_i'*SjInv*diff_i + trace(SjInv*V_i) + log(det(Sj)));
            end
        end

        objTrace(it) = obj;

        relChange = abs(prevObj - obj) / max(1, abs(prevObj));
        relTrace(it) = relChange;
        finalRelChange = relChange;

        currentOuterStateVec = [m_hat(:); logtheta(:); Sigma_comp{1}(:); Sigma_comp{2}(:); alpha(:); gamma(:)];
        [stateRel, prevOuterStateVec] = outer_state_rel_change_local( ...
            currentOuterStateVec, prevOuterStateVec);
        stateRelTrace(it) = stateRel;
        finalStateRelChange = stateRel;

        if outer_convergence_met_local(relTrace, stateRelTrace, it, opts)
            convergedFlag = true;
            break;
        end

        prevObj = obj;
    end

    Kx_final = ard_rbf_kernel(X, X, logtheta, opts.jitter);
    Ktheta_final = make_spd(kron(Kx_final, eye(p)), opts.jitter);

    fitInfo = struct();
    fitInfo.nIter = it;
    fitInfo.maxIter = opts.maxIter;
    fitInfo.converged = convergedFlag;
    fitInfo.hitMaxIter = it >= opts.maxIter && ~convergedFlag;
    fitInfo.finalRelChange = finalRelChange;
    fitInfo.finalStateRelChange = finalStateRelChange;
    fitInfo.outerConvergenceCriterion = get_outer_criterion_local(opts);
    fitInfo.objTrace = objTrace(1:it);
    fitInfo.stateRelTrace = stateRelTrace(1:it);
    fitInfo.relTrace = relTrace(1:it);
    fitInfo.thetaGradNormTrace = thetaGradNormTrace(1:it);
    fitInfo.thetaStepTrace = thetaStepTrace(1:it);
    fitInfo.thetaNStepsTrace = thetaNStepsTrace(1:it);
    fitInfo.thetaUpdateCount = thetaUpdateCount;
    fitInfo.thetaAcceptedCount = thetaAcceptedCount;
    fitInfo.thetaFallbackCount = thetaFallbackCount;
    fitInfo.thetaInnerConvergedCount = thetaInnerConvergedCount;
    fitInfo.thetaInnerHitMaxCount = thetaInnerHitMaxCount;
    fitInfo.thetaLineSearchFailureCount = thetaLineSearchFailureCount;
    fitInfo.thetaInnerFinalRelTrace = thetaInnerFinalRelTrace(1:it);
    fitInfo.thetaGradStep = opts.thetaGradStep;
    fitInfo.thetaGradMaxIter = opts.thetaGradMaxIter;

    model = struct( ...
        'X',X, ...
        'Y',Y, ...
        'n',n, ...
        'p',p, ...
        'mu_f',mu_f, ...
        'Sigma_f',Sigma_f, ...
        'm_hat',m_hat, ...
        'logtheta',logtheta, ...
        'Kinv',inv(Ktheta_final), ...
        'Sigma_comp',{Sigma_comp}, ...
        'alpha',alpha, ...
        'gamma',gamma, ...
        'fitInfo', fitInfo, ...
        'method', 'GMM-GPR-LRDiag');
end

function [Xtrain, Xtest, Ytrain, Ytest] = load_real_numeric_split_csv( ...
    filePath, nTrain, nTest, seed, outputCols, numOutputs)
%LOAD_REAL_NUMERIC_SPLIT_CSV
% Generic real-data CSV loader for robust GPR experiments.
%
% If outputCols is empty:
%   Y = last numOutputs columns
%   X = all previous columns
%
% If outputCols is provided:
%   Y = selected columns
%   X = all remaining columns
%
% X and Y are standardized using training statistics only.

    rng(seed);

    if ~exist(filePath, 'file')
        error('Real-data CSV file not found: %s', filePath);
    end

    T = readtable(filePath);

    % Convert table to numeric matrix.
    A = table2array(T);
    A = double(A);

    % Remove rows with missing or non-finite values.
    A = A(all(isfinite(A), 2), :);

    [N, D] = size(A);

    if D < 3
        error('Dataset must have at least 3 numeric columns. Found D = %d.', D);
    end

    if nTrain + nTest > N
        error('Requested nTrain+nTest = %d but dataset has only N = %d rows.', ...
            nTrain + nTest, N);
    end

    % ------------------------------------------------------------
    % Choose output columns
    % ------------------------------------------------------------
    if isempty(outputCols)

        if nargin < 6 || isempty(numOutputs)
            numOutputs = 3;
        end

        if numOutputs >= D
            error('numOutputs = %d must be smaller than total columns D = %d.', ...
                numOutputs, D);
        end

        outputCols = (D - numOutputs + 1):D;
    end

    outputCols = outputCols(:)';

    if any(outputCols < 1) || any(outputCols > D)
        error('Invalid outputCols. Dataset has only D = %d columns.', D);
    end

    inputCols = setdiff(1:D, outputCols);

    if isempty(inputCols)
        error('No input columns left after selecting outputCols.');
    end

    X = A(:, inputCols);
    Y = A(:, outputCols);

    % ------------------------------------------------------------
    % Random train/test split
    % ------------------------------------------------------------
    idx = randperm(N);

    idxTrain = idx(1:nTrain);
    idxTest  = idx(nTrain+1:nTrain+nTest);

    XtrainRaw = X(idxTrain, :);
    XtestRaw  = X(idxTest,  :);

    YtrainRaw = Y(idxTrain, :);
    YtestRaw  = Y(idxTest,  :);

    % ------------------------------------------------------------
    % Standardize X using train only
    % ------------------------------------------------------------
    muX = mean(XtrainRaw, 1);
    sdX = std(XtrainRaw, 0, 1);
    sdX(~isfinite(sdX) | sdX < 1e-12) = 1;

    Xtrain = (XtrainRaw - muX) ./ sdX;
    Xtest  = (XtestRaw  - muX) ./ sdX;

    % ------------------------------------------------------------
    % Standardize Y using train only
    % ------------------------------------------------------------
    muY = mean(YtrainRaw, 1);
    sdY = std(YtrainRaw, 0, 1);
    sdY(~isfinite(sdY) | sdY < 1e-12) = 1;

    Ytrain = (YtrainRaw - muY) ./ sdY;
    Ytest  = (YtestRaw  - muY) ./ sdY;

    fprintf('Loaded real split: file = %s\n', filePath);
    fprintf('  Raw finite rows = %d | total numeric cols = %d\n', N, D);
    fprintf('  X cols = %s\n', mat2str(inputCols));
    fprintf('  Y cols = %s\n', mat2str(outputCols));
    fprintf('  X dim = %d | Y dim = %d | train = %d | test = %d\n', ...
        size(Xtrain,2), size(Ytrain,2), size(Xtrain,1), size(Xtest,1));
end


function assert_nonoracle_information_boundary(optsMethod, methodName, fixedRcgprEpsilon)
%ASSERT_NONORACLE_INFORMATION_BOUNDARY
% Runtime guard against accidental delivery of experiment truth to any
% non-Oracle method. Fixed, predeclared robust-model hyperparameters (for
% example GMM component-initialization multipliers) are permitted; the
% realized/simulated contamination settings and Oracle-only quantities are not.

    forbiddenFields = { ...
        'pOut', 'p_out', 'pOutNow', 'pOutFixed', 'pOutList', ...
        'outlierProbability', 'outlierProb', 'outlierRate', ...
        'outlierStd', 'outlierVariance', 'outlierVar', ...
        'outlierScale', 'outlierSecondMoment', 'outlierMean', ...
        'outlierModel', 'isOutlier', 'isOutlierEntry', ...
        'SigmaTrue', 'sigma_true', 'inlierVar', ...
        'oracle_R', 'oracle_inlier_var', 'oracle_outlier_var', ...
        'oracleInlierObsVar', 'oracleOutlierObsVar', ...
        'oracleBestGuessSigma', 'oracleBestGuessInlierVar'};

    optionFields = fieldnames(optsMethod);
    forbiddenMask = ismember(lower(string(optionFields)), ...
        lower(string(forbiddenFields)));

    if any(forbiddenMask)
        leakedFields = strjoin(optionFields(forbiddenMask), ', ');
        error('%s received forbidden contamination/Oracle field(s): %s', ...
            methodName, leakedFields);
    end

    if ~isfield(optsMethod, 'rcgp_epsilon') || ...
            ~isscalar(optsMethod.rcgp_epsilon) || ...
            ~isfinite(optsMethod.rcgp_epsilon) || ...
            abs(optsMethod.rcgp_epsilon - fixedRcgprEpsilon) > 10*eps
        error(['%s must use the fixed, contamination-independent RCGPR ' ...
            'epsilon %.16g.'], methodName, fixedRcgprEpsilon);
    end
end
