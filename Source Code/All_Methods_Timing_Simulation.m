%% All_Methods_Extensive_Timing_Simulation
% Serial fitting-time scaling comparison for:
%   1. Oracle
%   2. ASOR-GPR
%   3. GMM-GPR
%   4. Student-t GPR
%   5. RCGPR
%   6. Standard GPR
%
% This script extends the completed ASOR-only timing experiment without
% changing its data-generating protocol:
%   nTrainList = [200 400 600 800 1000]
%   nMC        = 30 per training-set size
%   p_out      = 0.40 (generator only)
%   outliers   = N(0,10), where 10 is the variance (generator only)
%   nominal noise variance = 0.25 (generator and Oracle only)
%   shared non-oracle initial noise variance = 1
%   ASOR/GMM outer VB/EM upper limit = 1000
%   no parfor; one MATLAB computational thread
%
% Only model fitting is timed. Data generation, option construction,
% diagnostics, prediction, plotting, printing, and saving are outside the
% timed intervals.
%
% STRICT INFORMATION BOUNDARY
% ---------------------------
% Every non-oracle fitting call is made through fit_nonoracle_method(),
% whose arguments contain only:
%   - Xtrain,
%   - the contaminated Ytrain,
%   - fixed algorithm settings and fixed non-oracle initialization.
%
% No non-oracle method receives p_out, the realized contamination mask,
% the true nominal variance, the generating Gaussian outlier variance, or
% an Oracle component variance. The Oracle alone receives the entry-wise
% mask and the two true component residual variances.

clear; clc; close all;
rng(7, 'twister');

%% ========================================================================
% Enforce serial execution
% ========================================================================
if license('test', 'Distrib_Computing_Toolbox') == 1
    existingPool = gcp('nocreate');
    if ~isempty(existingPool)
        delete(existingPool);
    end
end

previousMaxThreads = maxNumCompThreads(1);
threadCleanup = onCleanup(@() maxNumCompThreads(previousMaxThreads)); %#ok<NASGU>

fprintf('\n============================================================\n');
fprintf('ALL-METHOD SERIAL SCALING EXPERIMENT\n');
fprintf('nTrain = 200,400,600,800,1000 | 30 MC runs per size\n');
fprintf('No parfor | one MATLAB computational thread\n');
fprintf('============================================================\n');

%% ========================================================================
% Paths
% ========================================================================
userRoot = 'C:\Users\majal';

% GPML is required by the Student-t GPR baseline.
gpml_path = 'C:\Users\majal\Downloads\gpml-matlab-v4.2-2018-06-11';

if ~exist(gpml_path, 'dir')
    error(['GPML directory not found:\n  %s\n' ...
        'Correct gpml_path before running the complete comparison.'], ...
        gpml_path);
end

addpath(genpath(gpml_path));
evalc('startup;');

runTag = datestr(now, 'yyyymmdd_HHMMSS');
outputDir = fullfile(userRoot, ...
    ['All_Methods_Serial_Scaling_n200_1000_MC30_MAXITER1000_' runTag]);

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

resultFile = fullfile(outputDir, ...
    'all_methods_serial_scaling_n200_1000_MC30_results.mat');
summaryCsvFile = fullfile(outputDir, ...
    'all_methods_serial_scaling_n200_1000_MC30_summary.csv');
summaryTxtFile = fullfile(outputDir, ...
    'all_methods_serial_scaling_n200_1000_MC30_summary.txt');

fprintf('\nResults will be saved in:\n  %s\n', outputDir);

%% ========================================================================
% Experiment settings -- identical to the ASOR-only timing experiment
% ========================================================================
d_x_fixed = 2;
p_y_fixed = 4;

nTrainList = [200 400 600 800 1000];
nMC = 30;
nN = numel(nTrainList);

% Generator-only contamination settings.
pOutFixed = 0.40;
outlierModel = 'gaussian';
outlierScale = sqrt(10.0);  % standard deviation; variance is 10

% Generator-only nominal noise setting.
sigma_true = 0.50;
inlierVar = sigma_true^2;

% Fixed, intentionally incorrect initialization shared by every
% non-oracle method.
sigma_init_wrong = 1.00;
sigma_asor_prior = sigma_init_wrong;

switch lower(outlierModel)
    case 'positive_uniform'
        outlierVar = outlierScale^2 / 12;
        outlierSecondMoment = outlierScale^2 / 3;
    case 'negative_uniform'
        outlierVar = outlierScale^2 / 12;
        outlierSecondMoment = outlierScale^2 / 3;
    case 'uniform'
        outlierVar = outlierScale^2 / 3;
        outlierSecondMoment = outlierScale^2 / 3;
    case 'gaussian'
        outlierVar = outlierScale^2;
        outlierSecondMoment = outlierScale^2;
    case 'positive_shift'
        outlierVar = 0;
        outlierSecondMoment = outlierScale^2;
    case 'negative_shift'
        outlierVar = 0;
        outlierSecondMoment = outlierScale^2;
    otherwise
        error('Unknown outlierModel = %s', outlierModel);
end

% These two quantities are Oracle-only. They are never inserted into the
% non-oracle options structure.
oracleInlierObsVar = inlierVar;
oracleOutlierObsVar = inlierVar + outlierSecondMoment;

dataMode = 'debug_grid';
trueFunctionKind = 'friedman_like';

fprintf('\nFixed scaling settings:\n');
fprintf('  d_x_fixed              = %d\n', d_x_fixed);
fprintf('  p_y_fixed              = %d\n', p_y_fixed);
fprintf('  nTrainList             = %s\n', mat2str(nTrainList));
fprintf('  nMC                    = %d\n', nMC);
fprintf('  pOutFixed [generator]  = %.2f\n', pOutFixed);
fprintf('  inlierVar [generator]  = %.4f\n', inlierVar);
fprintf('  outlierVar [generator] = %.4f\n', outlierVar);
fprintf('  sigma_init_wrong       = %.4f\n', sigma_init_wrong);

%% ========================================================================
% Fixed non-oracle algorithm settings
% ========================================================================
opts = struct();

% ASOR/GMM outer loop: unchanged from the ASOR diagnostic.
opts.maxIter = 1000;
opts.tol = 1e-5;
opts.outerConvergenceCriterion = 'obj_window3_max';

opts.init_signal_std = 1.0;
opts.init_lengthscale = 1.0;

opts.learnTheta = true;
opts.thetaBurnIn = 0;
opts.thetaUpdateEvery = 1;

opts.thetaOptimizer = 'gd_warmstart';
opts.thetaGradStep = 0.05;
opts.thetaGradMaxIter = 25;
opts.thetaMaxIter = opts.thetaGradMaxIter;
opts.thetaMaxFunEvals = 20;
opts.thetaGradArmijo = 1e-4;
opts.thetaGradMinStep = 1e-7;
opts.thetaGradVerbose = false;

opts.thetaInnerRelTol = 1e-5;
opts.thetaGradTol = 1e-5;
opts.thetaMinInnerIter = 2;

opts.thetaStepIncrease = 1.25;
opts.thetaGradStepMinStart = 1e-5;
opts.thetaGradStepMaxStart = opts.thetaGradStep;
opts.thetaMomentumBeta = 0.90;
opts.thetaAdamBeta1 = 0.90;
opts.thetaAdamBeta2 = 0.999;
opts.thetaAdamEps = 1e-8;

% Standard GPR and Oracle use their existing one-shot optimizer budget.
opts.fixedGPGradMaxIter = 60;

% ASOR parameters and priors.
opts.a = 1;
opts.theta0 = 0.5;
opts.A = 10;
opts.nu0_base_offset = 2;

nu0_scalar = 1 + opts.nu0_base_offset;
p_scalar = 1;
opts.S0_scale = (nu0_scalar + p_scalar + 1) * sigma_asor_prior^2;

robustScaleFactor = 20;
opts.init_b = opts.a * robustScaleFactor;
opts.B = (opts.A - 1) / opts.init_b;

opts.alpha0_sigma = 1.2;
opts.beta0_sigma = (opts.alpha0_sigma - 1) * sigma_init_wrong^2;
opts.asor_R_update = 'iw_mode_original';

% GMM initialization is fixed independently of the generated outliers.
% It is not calculated from pOutFixed, outlierScale, or outlierVar.
opts.gmm_outlier_init_multiplier = 10.0;
opts.gmm_nominal_init_multiplier = 1.0;

% RCGPR settings: unchanged from the comparison reference.
opts.rcgp_epsilon = 0.05;
opts.rcgp_useShrinkageTerm = true;
opts.rcgp_shrinkageConvention = 'section3';
opts.rcgp_maxIter = 100;
opts.rcgp_lbfgsMemory = 10;
opts.rcgp_gradTol = 1e-5;
opts.rcgp_stepTol = 1e-10;
opts.rcgp_armijo = 1e-4;
opts.rcgp_minStep = 1e-10;
opts.rcgp_verbose = false;

% Student-t GPML settings: unchanged from the comparison reference.
% opts.maxIter controls ASOR/GMM only; Student-t retains the method-specific
% GPML hyperparameter-optimization budget used in the supplied code.
opts.student_gpml_inference = 'VB';
opts.student_gpml_nu = 4;
opts.student_gpml_optIters = -60;

opts.jitter = 1e-6;
opts.minWeight = 1e-6;
opts.verbose = false;

fprintf('\nASOR/GMM stopping setup:\n');
fprintf('  outer maxIter             = %d\n', opts.maxIter);
fprintf('  outer tol                 = %.1e\n', opts.tol);
fprintf('  outer criterion           = %s\n', ...
    opts.outerConvergenceCriterion);
fprintf('  thetaGradMaxIter          = %d\n', opts.thetaGradMaxIter);
fprintf('  Student-t GPML budget     = %d\n', opts.student_gpml_optIters);
fprintf('  RCGPR L-BFGS maxIter      = %d\n', opts.rcgp_maxIter);
fprintf('  Standard/Oracle maxIter   = %d\n', opts.fixedGPGradMaxIter);

%% ========================================================================
% Methods and colors
% ========================================================================
methodNames = {'Oracle', 'ASOR-GPR', 'GMM-GPR', ...
    'Student-t GPR', 'RCGPR', 'Standard GPR'};
methodKeys = {'oracle', 'asor', 'gmm', 'student', 'rcgp', 'standard'};
nMethods = numel(methodNames);

IDX_ORACLE = 1;
IDX_ASOR = 2;
IDX_GMM = 3;
IDX_STUDENT = 4;
IDX_RCGP = 5;
IDX_STANDARD = 6;

colors = [
    0.25 0.25 0.25;
    35/255 139/255 69/255;
    180/255 60/255 120/255;
    0.75 0.10 0.10;
    20/255 120/255 180/255;
    220/255 120/255 20/255
];

%% ========================================================================
% Untimed JIT/GPML warm-up
% ========================================================================
% Warm-up uses reduced iteration budgets and is not part of any MC trial.
rng(987654, 'twister');
Xwarm = randn(30, d_x_fixed);
Ywarm = randn(30, p_y_fixed);

optsWarm = opts;
optsWarm.maxIter = 3;
optsWarm.thetaGradMaxIter = 1;
optsWarm.thetaMaxIter = 1;
optsWarm.fixedGPGradMaxIter = 2;
optsWarm.rcgp_maxIter = 2;
optsWarm.student_gpml_optIters = -2;
optsWarm.nu0 = p_y_fixed + optsWarm.nu0_base_offset;
optsWarm.init_R = sigma_init_wrong^2 * eye(p_y_fixed);
optsWarm.sigma_init_wrong = sigma_init_wrong;
optsWarm.init_lengthscale = 1.0;
optsWarm.init_signal_std = 1.0;
optsWarm.use_median_mean_init = false;

assert_nonoracle_information_boundary(optsWarm, sigma_init_wrong, p_y_fixed);

warmNonOracleKeys = {'asor', 'gmm', 'student', 'rcgp', 'standard'};
for kk = 1:numel(warmNonOracleKeys)
    try
        warmModel = fit_nonoracle_method( ...
            warmNonOracleKeys{kk}, Xwarm, Ywarm, optsWarm, ...
            sigma_init_wrong); %#ok<NASGU>
    catch ME
        warning('Untimed warm-up failed for %s: %s', ...
            warmNonOracleKeys{kk}, ME.message);
    end
end

warmMask = rand(size(Ywarm)) < 0.40;
optsOracleWarm = optsWarm;
optsOracleWarm.oracle_inlier_var = oracleInlierObsVar;
optsOracleWarm.oracle_outlier_var = oracleOutlierObsVar;

try
    warmOracle = independent_oracle_gmm_style_gpr_fit( ...
        Xwarm, Ywarm, warmMask, optsOracleWarm); %#ok<NASGU>
catch ME
    warning('Untimed Oracle warm-up failed: %s', ME.message);
end

clear Xwarm Ywarm warmMask optsWarm optsOracleWarm warmModel warmOracle
rng(7, 'twister');

%% ========================================================================
% Storage
% ========================================================================
% Dimensions: MC trial x nTrain case x method.
fitTime = nan(nMC, nN, nMethods);
fitSucceeded = false(nMC, nN, nMethods);
errorLog = strings(nMC, nN, nMethods);

% General iteration diagnostics. The meaning is method-specific:
% ASOR/GMM = outer VB/EM iterations;
% RCGPR/Standard/Oracle = L-BFGS iterations;
% Student-t = unavailable from the supplied GPML wrapper, hence NaN.
methodIterMedian = nan(nMC, nN, nMethods);
methodIterMax = nan(nMC, nN, nMethods);
methodIterSum = nan(nMC, nN, nMethods);

% Detailed ASOR/GMM outer-loop diagnostics.
ASOR_outerConverged = false(nMC, nN);
GMM_outerConverged = false(nMC, nN);
ASOR_hitMaxIter = false(nMC, nN);
GMM_hitMaxIter = false(nMC, nN);

ASOR_iterMedian = nan(nMC, nN);
GMM_iterMedian = nan(nMC, nN);
ASOR_iterMax = nan(nMC, nN);
GMM_iterMax = nan(nMC, nN);
ASOR_iterSum = nan(nMC, nN);
GMM_iterSum = nan(nMC, nN);

ASOR_finalRelMax = nan(nMC, nN);
GMM_finalRelMax = nan(nMC, nN);
ASOR_innerConvFrac = nan(nMC, nN);
GMM_innerConvFrac = nan(nMC, nN);
ASOR_innerHitCapFrac = nan(nMC, nN);
GMM_innerHitCapFrac = nan(nMC, nN);
ASOR_lineSearchFailFrac = nan(nMC, nN);
GMM_lineSearchFailFrac = nan(nMC, nN);

realizedOutlierEntryCount = zeros(nMC, nN);
realizedOutlierFrac = nan(nMC, nN);
ell0_store = nan(nMC, nN);
seed_store = zeros(nMC, nN);

completedCaseMask = false(1, nN);
caseCompletedTimestamp = strings(1, nN);
lastCompletedCaseIdx = 0;
lastCompletedNTrain = NaN;

settings = struct();
settings.purpose = 'Serial all-method fitting-time scaling experiment.';
settings.runTag = runTag;
settings.is_parallel = false;
settings.max_computational_threads = 1;
settings.d_x_fixed = d_x_fixed;
settings.p_y_fixed = p_y_fixed;
settings.nTrainList = nTrainList;
settings.nMC = nMC;
settings.pOutFixed_generator_only = pOutFixed;
settings.sigma_true_generator_only = sigma_true;
settings.inlierVar_generator_and_oracle_only = inlierVar;
settings.sigma_init_wrong_nonoracle = sigma_init_wrong;
settings.outlierModel_generator_only = outlierModel;
settings.outlierScale_generator_only = outlierScale;
settings.outlierVar_generator_only = outlierVar;
settings.outlierSecondMoment_generator_and_oracle_only = outlierSecondMoment;
settings.oracleInlierObsVar = oracleInlierObsVar;
settings.oracleOutlierObsVar = oracleOutlierObsVar;
settings.dataMode = dataMode;
settings.trueFunctionKind = trueFunctionKind;
settings.methodNames = methodNames;
settings.methodKeys = methodKeys;
settings.outerMaxIter_ASOR_GMM = opts.maxIter;
settings.outerTol_ASOR_GMM = opts.tol;
settings.outerConvergenceCriterion_ASOR_GMM = ...
    opts.outerConvergenceCriterion;
settings.student_gpml_optIters = opts.student_gpml_optIters;
settings.rcgp_maxIter = opts.rcgp_maxIter;
settings.fixedGPGradMaxIter = opts.fixedGPGradMaxIter;
settings.untimed_jit_warmup = true;
settings.timing_scope = 'fit only; excludes generation, diagnostics, prediction, plotting, and saving';
settings.nonoracle_boundary = [ ...
    "Non-oracle helper receives only Xtrain, contaminated Ytrain, fixed settings, and fixed initialization."; ...
    "No non-oracle method receives p_out, mask, true nominal variance, generated outlier variance, or Oracle variances."; ...
    "Only Oracle receives the entry-wise mask and true component residual variances." ...
];

%% ========================================================================
% Main serial scaling experiment
% ========================================================================
totalTimer = tic;

for nn = 1:nN

    nTrain = nTrainList(nn);
    p_y = p_y_fixed;
    d_x = d_x_fixed;

    fprintf('\n\n============================================================\n');
    fprintf('Scaling case %d/%d: nTrain = %d\n', nn, nN, nTrain);
    fprintf('30 MC runs | serial | p_out(generator only) = %.2f\n', ...
        pOutFixed);
    fprintf('============================================================\n');

    SigmaTrue = sigma_true^2 * eye(p_y);
    SigmaInitWrong = sigma_init_wrong^2 * eye(p_y);

    opts_dim = opts;
    opts_dim.nu0 = p_y + opts.nu0_base_offset;

    caseTimer = tic;

    for mc = 1:nMC

        fprintf('\n------------------------------------------------------------\n');
        fprintf('nTrain=%d | MC %d/%d\n', nTrain, mc, nMC);
        fprintf('------------------------------------------------------------\n');

        seedNow = 100000 + 1000*nn + mc;
        seed_store(mc, nn) = seedNow;
        rng(seedNow, 'twister');

        % Training-only generator: identical to the ASOR timing script.
        [Xtrain, YcleanTrain] = ...
            generate_synthetic_training_dataset_scaling( ...
            nTrain, d_x, p_y, dataMode, trueFunctionKind, ...
            1000*nn + mc);

        E = mvnrnd_local(zeros(p_y,1), SigmaTrue, nTrain);
        Ytrain = YcleanTrain + E;

        outlierNoiseAll = generate_outlier_noise( ...
            nTrain, p_y, outlierScale, outlierModel);
        isOutlierEntry = rand(nTrain, p_y) < pOutFixed;
        Ytrain(isOutlierEntry) = Ytrain(isOutlierEntry) + ...
            outlierNoiseAll(isOutlierEntry);

        realizedOutlierEntryCount(mc, nn) = sum(isOutlierEntry(:));
        realizedOutlierFrac(mc, nn) = ...
            realizedOutlierEntryCount(mc, nn) / numel(isOutlierEntry);

        % Build the clean non-oracle options object. No Oracle fields are
        % placed in opts_dim or opts_nonoracle.
        opts_nonoracle = opts_dim;
        opts_nonoracle.init_R = SigmaInitWrong;
        opts_nonoracle.sigma_init_wrong = sigma_init_wrong;

        ell0 = 0.20 * median_pairwise_distance(Xtrain);
        if ~isfinite(ell0) || ell0 <= 0
            ell0 = opts.init_lengthscale;
        end
        ell0 = max(ell0, 1e-2);

        ell0_store(mc, nn) = ell0;
        opts_nonoracle.init_lengthscale = ell0;
        opts_nonoracle.init_signal_std = 1.0;
        opts_nonoracle.use_median_mean_init = false;

        assert_nonoracle_information_boundary( ...
            opts_nonoracle, sigma_init_wrong, p_y);

        % ------------------------------------------------------------
        % Fit all non-oracle methods in the same order as the supplied
        % comparison code. Every call uses the same Xtrain and Ytrain.
        % ------------------------------------------------------------
        nonOracleIndices = [IDX_ASOR IDX_GMM IDX_STUDENT IDX_RCGP IDX_STANDARD];

        for jj = 1:numel(nonOracleIndices)
            mm = nonOracleIndices(jj);
            methodKeyNow = methodKeys{mm};

            try
                tStart = tic;
                modelNow = fit_nonoracle_method( ...
                    methodKeyNow, Xtrain, Ytrain, opts_nonoracle, ...
                    sigma_init_wrong);
                elapsedNow = toc(tStart);

                assert_model_fit_success(modelNow, methodKeyNow);

                fitTime(mc, nn, mm) = elapsedNow;
                fitSucceeded(mc, nn, mm) = true;

                [iterMedNow, iterMaxNow, iterSumNow] = ...
                    collect_method_iterations(modelNow, methodKeyNow);
                methodIterMedian(mc, nn, mm) = iterMedNow;
                methodIterMax(mc, nn, mm) = iterMaxNow;
                methodIterSum(mc, nn, mm) = iterSumNow;

                if mm == IDX_ASOR
                    diagNow = collect_independent_fit_diagnostics_cap(modelNow);
                    ASOR_outerConverged(mc,nn) = diagNow.allConverged;
                    ASOR_hitMaxIter(mc,nn) = diagNow.anyHitMaxIter;
                    ASOR_iterMedian(mc,nn) = diagNow.iterMedian;
                    ASOR_iterMax(mc,nn) = diagNow.iterMax;
                    if all(isfinite(diagNow.nIter))
                        ASOR_iterSum(mc,nn) = sum(diagNow.nIter);
                    end
                    ASOR_finalRelMax(mc,nn) = diagNow.finalRelMax;
                    ASOR_innerConvFrac(mc,nn) = diagNow.thetaInnerConvFrac;
                    ASOR_innerHitCapFrac(mc,nn) = ...
                        diagNow.thetaInnerHitCapFrac;
                    ASOR_lineSearchFailFrac(mc,nn) = ...
                        diagNow.thetaLineSearchFailFrac;

                elseif mm == IDX_GMM
                    diagNow = collect_independent_fit_diagnostics_cap(modelNow);
                    GMM_outerConverged(mc,nn) = diagNow.allConverged;
                    GMM_hitMaxIter(mc,nn) = diagNow.anyHitMaxIter;
                    GMM_iterMedian(mc,nn) = diagNow.iterMedian;
                    GMM_iterMax(mc,nn) = diagNow.iterMax;
                    if all(isfinite(diagNow.nIter))
                        GMM_iterSum(mc,nn) = sum(diagNow.nIter);
                    end
                    GMM_finalRelMax(mc,nn) = diagNow.finalRelMax;
                    GMM_innerConvFrac(mc,nn) = diagNow.thetaInnerConvFrac;
                    GMM_innerHitCapFrac(mc,nn) = ...
                        diagNow.thetaInnerHitCapFrac;
                    GMM_lineSearchFailFrac(mc,nn) = ...
                        diagNow.thetaLineSearchFailFrac;
                end

                fprintf('  %-16s fit time = %10.4f s', ...
                    methodNames{mm}, elapsedNow);
                if isfinite(iterMaxNow)
                    fprintf(' | iteration max = %.0f', iterMaxNow);
                end
                fprintf('\n');

                clear modelNow diagNow

            catch ME
                fitTime(mc, nn, mm) = NaN;
                fitSucceeded(mc, nn, mm) = false;
                errorLog(mc, nn, mm) = string(ME.message);
                warning('%s failed at nTrain=%d, MC=%d: %s', ...
                    methodNames{mm}, nTrain, mc, ME.message);
            end
        end

        % ------------------------------------------------------------
        % Oracle-only call. The realized mask and true component residual
        % variances enter only this separate options structure and call.
        % ------------------------------------------------------------
        opts_oracle = opts_nonoracle;
        opts_oracle.oracle_inlier_var = oracleInlierObsVar;
        opts_oracle.oracle_outlier_var = oracleOutlierObsVar;

        try
            tStart = tic;
            modelOracle = independent_oracle_gmm_style_gpr_fit( ...
                Xtrain, Ytrain, isOutlierEntry, opts_oracle);
            elapsedOracle = toc(tStart);

            assert_model_fit_success(modelOracle, 'oracle');

            fitTime(mc, nn, IDX_ORACLE) = elapsedOracle;
            fitSucceeded(mc, nn, IDX_ORACLE) = true;

            [iterMedNow, iterMaxNow, iterSumNow] = ...
                collect_method_iterations(modelOracle, 'oracle');
            methodIterMedian(mc, nn, IDX_ORACLE) = iterMedNow;
            methodIterMax(mc, nn, IDX_ORACLE) = iterMaxNow;
            methodIterSum(mc, nn, IDX_ORACLE) = iterSumNow;

            fprintf('  %-16s fit time = %10.4f s', ...
                methodNames{IDX_ORACLE}, elapsedOracle);
            if isfinite(iterMaxNow)
                fprintf(' | optimizer iter max = %.0f', iterMaxNow);
            end
            fprintf('\n');

            clear modelOracle

        catch ME
            fitTime(mc, nn, IDX_ORACLE) = NaN;
            fitSucceeded(mc, nn, IDX_ORACLE) = false;
            errorLog(mc, nn, IDX_ORACLE) = string(ME.message);
            warning('Oracle failed at nTrain=%d, MC=%d: %s', ...
                nTrain, mc, ME.message);
        end

        fprintf('  realized outlier fraction = %.4f\n', ...
            realizedOutlierFrac(mc, nn));
        fprintf('  ASOR outer conv/hit-max    = %d / %d\n', ...
            ASOR_outerConverged(mc,nn), ASOR_hitMaxIter(mc,nn));
        fprintf('  GMM outer conv/hit-max     = %d / %d\n', ...
            GMM_outerConverged(mc,nn), GMM_hitMaxIter(mc,nn));

        % Checkpoint after every completed MC trial. No fitted model is
        % stored, which keeps the checkpoint compact.
        save(resultFile, ...
            'fitTime', 'fitSucceeded', 'errorLog', ...
            'methodIterMedian', 'methodIterMax', 'methodIterSum', ...
            'ASOR_outerConverged', 'GMM_outerConverged', ...
            'ASOR_hitMaxIter', 'GMM_hitMaxIter', ...
            'ASOR_iterMedian', 'GMM_iterMedian', ...
            'ASOR_iterMax', 'GMM_iterMax', ...
            'ASOR_iterSum', 'GMM_iterSum', ...
            'ASOR_finalRelMax', 'GMM_finalRelMax', ...
            'ASOR_innerConvFrac', 'GMM_innerConvFrac', ...
            'ASOR_innerHitCapFrac', 'GMM_innerHitCapFrac', ...
            'ASOR_lineSearchFailFrac', 'GMM_lineSearchFailFrac', ...
            'realizedOutlierEntryCount', 'realizedOutlierFrac', ...
            'ell0_store', 'seed_store', ...
            'completedCaseMask', 'caseCompletedTimestamp', ...
            'lastCompletedCaseIdx', 'lastCompletedNTrain', ...
            'nTrainList', 'nMC', 'methodNames', 'methodKeys', ...
            'settings', 'opts', '-v7');
    end

    % ------------------------------------------------------------
    % Complete and save the current nTrain case
    % ------------------------------------------------------------
    completedCaseMask(nn) = true;
    lastCompletedCaseIdx = nn;
    lastCompletedNTrain = nTrain;
    caseCompletedTimestamp(nn) = ...
        string(datestr(now, 'yyyy-mm-dd HH:MM:SS'));

    caseElapsedSeconds = toc(caseTimer);
    caseCsvFile = fullfile(outputDir, ...
        sprintf('case_%02d_nTrain_%d_timing_summary.csv', nn, nTrain));

    write_case_runtime_csv(caseCsvFile, nTrain, methodNames, ...
        squeeze(fitTime(:,nn,:)), squeeze(fitSucceeded(:,nn,:)), ...
        squeeze(methodIterMedian(:,nn,:)), ...
        squeeze(methodIterMax(:,nn,:)), ...
        squeeze(methodIterSum(:,nn,:)));

    fprintf('\nCompleted nTrain=%d in %.2f s of wall-clock experiment time.\n', ...
        nTrain, caseElapsedSeconds);
    for mm = 1:nMethods
        fprintf('  %-16s median fit time = %.6g s | success rate = %.3f\n', ...
            methodNames{mm}, ...
            median(fitTime(:,nn,mm), 'omitnan'), ...
            mean(fitSucceeded(:,nn,mm)));
    end

    % Update and save the figures immediately after each completed size.
    update_all_scaling_figures(fitTime, ASOR_iterMax, GMM_iterMax, ...
        nTrainList, completedCaseMask, methodNames, colors, outputDir);

    save(resultFile, ...
        'fitTime', 'fitSucceeded', 'errorLog', ...
        'methodIterMedian', 'methodIterMax', 'methodIterSum', ...
        'ASOR_outerConverged', 'GMM_outerConverged', ...
        'ASOR_hitMaxIter', 'GMM_hitMaxIter', ...
        'ASOR_iterMedian', 'GMM_iterMedian', ...
        'ASOR_iterMax', 'GMM_iterMax', ...
        'ASOR_iterSum', 'GMM_iterSum', ...
        'ASOR_finalRelMax', 'GMM_finalRelMax', ...
        'ASOR_innerConvFrac', 'GMM_innerConvFrac', ...
        'ASOR_innerHitCapFrac', 'GMM_innerHitCapFrac', ...
        'ASOR_lineSearchFailFrac', 'GMM_lineSearchFailFrac', ...
        'realizedOutlierEntryCount', 'realizedOutlierFrac', ...
        'ell0_store', 'seed_store', ...
        'completedCaseMask', 'caseCompletedTimestamp', ...
        'lastCompletedCaseIdx', 'lastCompletedNTrain', ...
        'nTrainList', 'nMC', 'methodNames', 'methodKeys', ...
        'settings', 'opts', '-v7');
end

%% ========================================================================
% Final summaries and empirical scaling exponents
% ========================================================================
medianTime = nan(nN, nMethods);
meanTime = nan(nN, nMethods);
stdTime = nan(nN, nMethods);
successRate = nan(nN, nMethods);
runtimeExponent = nan(1, nMethods);
runtimeR2 = nan(1, nMethods);

for nn = 1:nN
    for mm = 1:nMethods
        medianTime(nn,mm) = median(fitTime(:,nn,mm), 'omitnan');
        meanTime(nn,mm) = mean(fitTime(:,nn,mm), 'omitnan');
        stdTime(nn,mm) = std(fitTime(:,nn,mm), 0, 'omitnan');
        successRate(nn,mm) = mean(fitSucceeded(:,nn,mm));
    end
end

for mm = 1:nMethods
    [runtimeExponent(mm), runtimeR2(mm)] = ...
        fit_loglog_power_law_scaling(nTrainList, medianTime(:,mm)');
end

medianASORIterSum = median(ASOR_iterSum, 1, 'omitnan');
medianGMMIterSum = median(GMM_iterSum, 1, 'omitnan');

[ASOR_iterNormalizedExponent, ASOR_iterNormalizedR2] = ...
    fit_loglog_power_law_scaling(nTrainList, ...
    medianTime(:,IDX_ASOR)' ./ medianASORIterSum);

[GMM_iterNormalizedExponent, GMM_iterNormalizedR2] = ...
    fit_loglog_power_law_scaling(nTrainList, ...
    medianTime(:,IDX_GMM)' ./ medianGMMIterSum);

summaryTable = build_long_summary_table(nTrainList, methodNames, ...
    medianTime, meanTime, stdTime, successRate, ...
    methodIterMedian, methodIterMax, methodIterSum, ...
    runtimeExponent, runtimeR2);
writetable(summaryTable, summaryCsvFile);

fid = fopen(summaryTxtFile, 'w');
if fid < 0
    warning('Could not open summary text file: %s', summaryTxtFile);
else
    fprintf(fid, 'ALL-METHOD SERIAL SCALING SUMMARY\n');
    fprintf(fid, '=================================\n');
    fprintf(fid, 'nTrainList = %s\n', mat2str(nTrainList));
    fprintf(fid, 'nMC = %d\n', nMC);
    fprintf(fid, 'one MATLAB computational thread; no parfor\n');
    fprintf(fid, 'ASOR/GMM outer maxIter = %d\n', opts.maxIter);
    fprintf(fid, 'p_out = %.2f [generator only]\n', pOutFixed);
    fprintf(fid, 'outlier model = N(0,10) [generator only]\n');
    fprintf(fid, 'non-oracle initial variance = %.6g\n\n', ...
        sigma_init_wrong^2);

    for mm = 1:nMethods
        fprintf(fid, '%s\n', methodNames{mm});
        fprintf(fid, '  median times = %s\n', ...
            mat2str(medianTime(:,mm)', 8));
        fprintf(fid, '  success rates = %s\n', ...
            mat2str(successRate(:,mm)', 5));
        fprintf(fid, '  raw median-time exponent = %.8f\n', ...
            runtimeExponent(mm));
        fprintf(fid, '  log-log R^2 = %.8f\n\n', runtimeR2(mm));
    end

    fprintf(fid, 'ASOR iteration-normalized exponent = %.8f (R^2=%.8f)\n', ...
        ASOR_iterNormalizedExponent, ASOR_iterNormalizedR2);
    fprintf(fid, 'GMM iteration-normalized exponent = %.8f (R^2=%.8f)\n', ...
        GMM_iterNormalizedExponent, GMM_iterNormalizedR2);
    fclose(fid);
end

update_all_scaling_figures(fitTime, ASOR_iterMax, GMM_iterMax, ...
    nTrainList, completedCaseMask, methodNames, colors, outputDir);

totalElapsedSeconds = toc(totalTimer);

save(resultFile, ...
    'fitTime', 'fitSucceeded', 'errorLog', ...
    'methodIterMedian', 'methodIterMax', 'methodIterSum', ...
    'ASOR_outerConverged', 'GMM_outerConverged', ...
    'ASOR_hitMaxIter', 'GMM_hitMaxIter', ...
    'ASOR_iterMedian', 'GMM_iterMedian', ...
    'ASOR_iterMax', 'GMM_iterMax', ...
    'ASOR_iterSum', 'GMM_iterSum', ...
    'ASOR_finalRelMax', 'GMM_finalRelMax', ...
    'ASOR_innerConvFrac', 'GMM_innerConvFrac', ...
    'ASOR_innerHitCapFrac', 'GMM_innerHitCapFrac', ...
    'ASOR_lineSearchFailFrac', 'GMM_lineSearchFailFrac', ...
    'realizedOutlierEntryCount', 'realizedOutlierFrac', ...
    'ell0_store', 'seed_store', ...
    'completedCaseMask', 'caseCompletedTimestamp', ...
    'lastCompletedCaseIdx', 'lastCompletedNTrain', ...
    'medianTime', 'meanTime', 'stdTime', 'successRate', ...
    'runtimeExponent', 'runtimeR2', ...
    'medianASORIterSum', 'medianGMMIterSum', ...
    'ASOR_iterNormalizedExponent', 'ASOR_iterNormalizedR2', ...
    'GMM_iterNormalizedExponent', 'GMM_iterNormalizedR2', ...
    'summaryTable', 'totalElapsedSeconds', ...
    'nTrainList', 'nMC', 'methodNames', 'methodKeys', ...
    'settings', 'opts', '-v7');

fprintf('\n============================================================\n');
fprintf('FINAL ALL-METHOD SERIAL SCALING SUMMARY\n');
fprintf('============================================================\n');
disp(summaryTable);

fprintf('\nEmpirical raw median-time exponents:\n');
for mm = 1:nMethods
    fprintf('  %-16s alpha = %.4f | log-log R^2 = %.4f\n', ...
        methodNames{mm}, runtimeExponent(mm), runtimeR2(mm));
end

fprintf('\nIteration-normalized exponents:\n');
fprintf('  ASOR-GPR alpha = %.4f | R^2 = %.4f\n', ...
    ASOR_iterNormalizedExponent, ASOR_iterNormalizedR2);
fprintf('  GMM-GPR  alpha = %.4f | R^2 = %.4f\n', ...
    GMM_iterNormalizedExponent, GMM_iterNormalizedR2);

fprintf('\nTotal experiment wall time: %.2f s (%.2f h)\n', ...
    totalElapsedSeconds, totalElapsedSeconds/3600);
fprintf('Final MAT file:\n  %s\n', resultFile);
fprintf('Final CSV summary:\n  %s\n', summaryCsvFile);
fprintf('Final text summary:\n  %s\n', summaryTxtFile);
fprintf('============================================================\n');

%% ========================================================================
% Timing-experiment helper functions
% ========================================================================

function model = fit_nonoracle_method(methodKey, X, Y, optsMethod, sigmaInitWrong)
%FIT_NONORACLE_METHOD Enforces the information boundary by construction.
% This local function has no mask, p_out, true variance, or generated
% outlier-scale argument.

    assert_nonoracle_information_boundary( ...
        optsMethod, sigmaInitWrong, size(Y,2));

    switch lower(methodKey)
        case 'asor'
            model = independent_asor_gpr_fit_lrdiag(X, Y, optsMethod);
        case 'gmm'
            model = independent_gmm_gpr_fit_lrdiag(X, Y, optsMethod);
        case 'student'
            model = studentt_gpml_mogp_fit(X, Y, optsMethod);
        case 'rcgp'
            model = independent_rcgp_gpr_fit(X, Y, optsMethod);
        case 'standard'
            model = independent_standard_gpr_fit(X, Y, optsMethod);
        otherwise
            error('Unknown non-oracle method key: %s', methodKey);
    end
end


function assert_nonoracle_information_boundary(optsMethod, sigmaInitWrong, p)
%ASSERT_NONORACLE_INFORMATION_BOUNDARY Rejects generator/Oracle fields.

    forbiddenFields = { ...
        'pOut', 'p_out', 'pOutFixed', 'pOutList', ...
        'outlierProbability', 'outlierProb', 'outlierRate', ...
        'outlierStd', 'outlierVariance', 'outlierVar', ...
        'outlierScale', 'outlierSecondMoment', 'outlierMean', ...
        'outlierModel', 'isOutlier', 'isOutlierEntry', ...
        'contaminationMask', 'contaminationIndicators', ...
        'sigma_true', 'inlierVar', 'SigmaTrue', ...
        'oracle_R', 'oracle_inlier_var', 'oracle_outlier_var'};

    presentFields = fieldnames(optsMethod);
    leaked = intersect(presentFields, forbiddenFields, 'stable');

    if ~isempty(leaked)
        error('Non-oracle options contain forbidden field(s): %s', ...
            strjoin(leaked, ', '));
    end

    if ~isfield(optsMethod, 'init_R')
        error('Every non-oracle method must receive the fixed init_R.');
    end

    expectedR = sigmaInitWrong^2 * eye(p);
    actualR = optsMethod.init_R;

    if ~isequal(size(actualR), size(expectedR)) || ...
            norm(actualR - expectedR, 'fro') > ...
            1e-12 * max(1, norm(expectedR, 'fro'))
        error(['Non-oracle init_R differs from the fixed shared ' ...
            'initialization.']);
    end

    if isfield(optsMethod, 'sigma_init_wrong') && ...
            abs(optsMethod.sigma_init_wrong - sigmaInitWrong) > 1e-12
        error('sigma_init_wrong is inconsistent with the fixed initialization.');
    end
end


function assert_model_fit_success(model, methodKey)
%ASSERT_MODEL_FIT_SUCCESS Prevents silent fallback timings from being valid.

    if ~isstruct(model) || ~isfield(model, 'output') || isempty(model.output)
        error('%s did not return a valid independent-output model.', methodKey);
    end

    if strcmpi(methodKey, 'student')
        for q = 1:numel(model.output)
            outq = model.output{q};
            if ~isfield(outq, 'ok') || ~outq.ok
                error('Student-t GPML failed or used fallback on output %d.', q);
            end
        end
    end
end


function [iterMedian, iterMax, iterSum] = ...
        collect_method_iterations(model, methodKey)
%COLLECT_METHOD_ITERATIONS Extracts method-specific iteration counts.

    p = numel(model.output);
    iterByOutput = nan(p,1);

    for q = 1:p
        outq = model.output{q};

        switch lower(methodKey)
            case {'asor', 'gmm'}
                if isfield(outq, 'fitInfo') && ...
                        isfield(outq.fitInfo, 'nIter')
                    iterByOutput(q) = outq.fitInfo.nIter;
                end

            case {'rcgp', 'standard'}
                if isfield(outq, 'fitInfo') && ...
                        isfield(outq.fitInfo, 'iterations')
                    iterByOutput(q) = outq.fitInfo.iterations;
                end

            case 'oracle'
                if isfield(outq, 'oracleInfo') && ...
                        isfield(outq.oracleInfo, 'iterations')
                    iterByOutput(q) = outq.oracleInfo.iterations;
                end

            case 'student'
                % The supplied GPML minimize wrapper does not expose a
                % reliable iteration count. Leave these values as NaN.
        end
    end

    good = isfinite(iterByOutput);
    if any(good)
        iterMedian = median(iterByOutput(good));
        iterMax = max(iterByOutput(good));
        iterSum = sum(iterByOutput(good));
    else
        iterMedian = NaN;
        iterMax = NaN;
        iterSum = NaN;
    end
end


function [Xtrain, YcleanTrain] = ...
        generate_synthetic_training_dataset_scaling( ...
        nTrain, d_x, p_y, dataMode, trueFunctionKind, mc)
%GENERATE_SYNTHETIC_TRAINING_DATASET_SCALING
% Training-only generator copied from the ASOR serial timing protocol.

    switch lower(dataMode)
        case 'debug_grid'
            if d_x == 1
                Xtrain = linspace(0,20,nTrain)';
            else
                rng(2000 + mc, 'twister');
                Xtrain = -3 + 6*lhsdesign_local(nTrain, d_x);
            end

        case 'random_box'
            rng(2000 + mc, 'twister');
            Xtrain = -3 + 6*rand(nTrain, d_x);

        otherwise
            error('Unknown dataMode: %s', dataMode);
    end

    YcleanTrain = true_multioutput_function( ...
        Xtrain, p_y, trueFunctionKind);

    trainMean = mean(YcleanTrain,1);
    trainStd = std(YcleanTrain,0,1);
    trainStd(~isfinite(trainStd) | trainStd < 1e-8) = 1;
    YcleanTrain = (YcleanTrain - trainMean) ./ trainStd;
end


function write_case_runtime_csv(filename, nTrain, methodNames, ...
        caseTimes, caseSuccess, caseIterMed, caseIterMax, caseIterSum)

    fid = fopen(filename, 'w');
    if fid < 0
        warning('Could not open case CSV file: %s', filename);
        return;
    end

    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, ['nTrain,Method,MedianTime_s,MeanTime_s,StdTime_s,' ...
        'SuccessRate,MedianMethodIterMedian,MedianMethodIterMax,' ...
        'MedianMethodIterSum\n']);

    for mm = 1:numel(methodNames)
        fprintf(fid, '%d,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n', ...
            nTrain, methodNames{mm}, ...
            median(caseTimes(:,mm), 'omitnan'), ...
            mean(caseTimes(:,mm), 'omitnan'), ...
            std(caseTimes(:,mm), 0, 'omitnan'), ...
            mean(caseSuccess(:,mm)), ...
            median(caseIterMed(:,mm), 'omitnan'), ...
            median(caseIterMax(:,mm), 'omitnan'), ...
            median(caseIterSum(:,mm), 'omitnan'));
    end
end


function T = build_long_summary_table(nTrainList, methodNames, ...
        medianTime, meanTime, stdTime, successRate, ...
        methodIterMedian, methodIterMax, methodIterSum, ...
        runtimeExponent, runtimeR2)

    nN = numel(nTrainList);
    nMethods = numel(methodNames);
    nRows = nN*nMethods;

    nTrainCol = zeros(nRows,1);
    methodCol = strings(nRows,1);
    medianCol = nan(nRows,1);
    meanCol = nan(nRows,1);
    stdCol = nan(nRows,1);
    successCol = nan(nRows,1);
    iterMedCol = nan(nRows,1);
    iterMaxCol = nan(nRows,1);
    iterSumCol = nan(nRows,1);
    exponentCol = nan(nRows,1);
    r2Col = nan(nRows,1);

    rr = 0;
    for nn = 1:nN
        for mm = 1:nMethods
            rr = rr + 1;
            nTrainCol(rr) = nTrainList(nn);
            methodCol(rr) = string(methodNames{mm});
            medianCol(rr) = medianTime(nn,mm);
            meanCol(rr) = meanTime(nn,mm);
            stdCol(rr) = stdTime(nn,mm);
            successCol(rr) = successRate(nn,mm);
            iterMedCol(rr) = median( ...
                methodIterMedian(:,nn,mm), 'omitnan');
            iterMaxCol(rr) = median( ...
                methodIterMax(:,nn,mm), 'omitnan');
            iterSumCol(rr) = median( ...
                methodIterSum(:,nn,mm), 'omitnan');
            exponentCol(rr) = runtimeExponent(mm);
            r2Col(rr) = runtimeR2(mm);
        end
    end

    T = table(nTrainCol, methodCol, medianCol, meanCol, stdCol, ...
        successCol, iterMedCol, iterMaxCol, iterSumCol, ...
        exponentCol, r2Col, ...
        'VariableNames', {'nTrain', 'Method', 'MedianTime_s', ...
        'MeanTime_s', 'StdTime_s', 'SuccessRate', ...
        'MedianMethodIterMedian', 'MedianMethodIterMax', ...
        'MedianMethodIterSum', 'RawRuntimeExponent', 'LogLogR2'});
end


function [alpha, rSquared] = fit_loglog_power_law_scaling(nValues, yValues)
    nValues = double(nValues(:));
    yValues = double(yValues(:));

    good = isfinite(nValues) & isfinite(yValues) & ...
        nValues > 0 & yValues > 0;

    if sum(good) < 2
        alpha = NaN;
        rSquared = NaN;
        return;
    end

    x = log(nValues(good));
    y = log(yValues(good));
    coeff = polyfit(x, y, 1);
    alpha = coeff(1);
    yHat = polyval(coeff, x);
    ssRes = sum((y-yHat).^2);
    ssTot = sum((y-mean(y)).^2);

    if ssTot <= eps
        rSquared = NaN;
    else
        rSquared = 1 - ssRes/ssTot;
    end
end


function update_all_scaling_figures(fitTime, ASOR_iterMax, GMM_iterMax, ...
        nTrainList, completedCaseMask, methodNames, colors, outputDir)

    completedIdx = find(completedCaseMask);
    if isempty(completedIdx)
        return;
    end

    dataNow = fitTime(:,completedIdx,:);
    nNow = nTrainList(completedIdx);

    % Linear-scale runtime box plots.
    fig1 = figure(101);
    clf(fig1);
    set(fig1, 'Color', 'w', 'Position', [40 60 1800 720]);
    draw_grouped_runtime_boxplots(dataNow, nNow, methodNames, colors, 'linear');
    save_scaling_figure(fig1, fullfile(outputDir, ...
        'All_methods_fit_time_boxplot_linear'));

    % Log-scale runtime box plots reveal all methods when runtimes differ
    % by orders of magnitude.
    fig2 = figure(102);
    clf(fig2);
    set(fig2, 'Color', 'w', 'Position', [60 80 1800 720]);
    draw_grouped_runtime_boxplots(dataNow, nNow, methodNames, colors, 'log');
    save_scaling_figure(fig2, fullfile(outputDir, ...
        'All_methods_fit_time_boxplot_log'));

    % Median runtime scaling on log-log axes.
    fig3 = figure(103);
    clf(fig3);
    set(fig3, 'Color', 'w', 'Position', [80 100 1500 760]);
    hold on;

    nMethods = numel(methodNames);
    for mm = 1:nMethods
        med = nan(1,numel(completedIdx));
        for kk = 1:numel(completedIdx)
            med(kk) = median(dataNow(:,kk,mm), 'omitnan');
        end
        loglog(nNow, med, '-o', 'Color', colors(mm,:), ...
            'MarkerFaceColor', colors(mm,:), 'LineWidth', 2.0, ...
            'MarkerSize', 8, 'DisplayName', methodNames{mm});
    end

    xlabel('$n_{\mathrm{train}}$', 'Interpreter', 'latex', 'FontSize', 26);
    ylabel('Median model fitting time (s)', ...
        'Interpreter', 'latex', 'FontSize', 25);
    set(gca, 'FontSize', 22, 'LineWidth', 1.3, ...
        'TickLabelInterpreter', 'latex');
    grid on;
    box on;
    legend('Location', 'northwest', 'FontSize', 16, ...
        'Interpreter', 'latex');
    hold off;
    drawnow;
    save_scaling_figure(fig3, fullfile(outputDir, ...
        'All_methods_median_fit_time_loglog'));

    % ASOR/GMM outer VB/EM iteration counts. Other methods use different
    % optimizers, so their counts are not mixed into this figure.
    fig4 = figure(104);
    clf(fig4);
    set(fig4, 'Color', 'w', 'Position', [100 120 1500 760]);

    iterCell = {ASOR_iterMax(:,completedIdx), ...
        GMM_iterMax(:,completedIdx)};
    iterNames = {'ASOR-GPR', 'GMM-GPR'};
    iterColors = [colors(2,:); colors(3,:)];
    draw_grouped_iteration_boxplots(iterCell, nNow, iterNames, iterColors);
    save_scaling_figure(fig4, fullfile(outputDir, ...
        'ASOR_GMM_outer_iteration_boxplot'));
end


function draw_grouped_runtime_boxplots(data, nTrainValues, ...
        methodNames, colors, yScaleMode)

    [~, nCases, nMethods] = size(data);
    centers = 1:nCases;
    offsets = linspace(-0.34, 0.34, nMethods);
    boxWidth = min(0.11, 0.72/max(nMethods,1));

    hold on;
    for mm = 1:nMethods
        for nn = 1:nCases
            y = data(:,nn,mm);
            y = y(isfinite(y) & y > 0);
            if isempty(y)
                continue;
            end
            x = (centers(nn) + offsets(mm))*ones(size(y));
            boxchart(x, y, ...
                'BoxFaceColor', colors(mm,:), ...
                'MarkerColor', colors(mm,:), ...
                'BoxWidth', boxWidth);
        end
    end

    legendHandles = gobjects(nMethods,1);
    for mm = 1:nMethods
        legendHandles(mm) = plot(nan, nan, 's', ...
            'MarkerFaceColor', colors(mm,:), ...
            'MarkerEdgeColor', colors(mm,:), ...
            'MarkerSize', 10, 'LineStyle', 'none');
    end

    xticks(centers);
    xticklabels(string(nTrainValues));
    xlim([0.5 nCases+0.5]);
    set(gca, 'YScale', yScaleMode, 'FontSize', 21, ...
        'LineWidth', 1.3, 'TickLabelInterpreter', 'latex');
    xlabel('$n_{\mathrm{train}}$', 'Interpreter', 'latex', 'FontSize', 27);
    ylabel('Model fitting time (s)', ...
        'Interpreter', 'latex', 'FontSize', 25);
    legend(legendHandles, methodNames, 'Location', 'northwest', ...
        'NumColumns', 3, 'FontSize', 15, 'Interpreter', 'latex');
    grid off;
    box on;
    hold off;
    drawnow;
end


function draw_grouped_iteration_boxplots(iterCell, nTrainValues, ...
        methodNames, colors)

    nMethods = numel(iterCell);
    nCases = numel(nTrainValues);
    centers = 1:nCases;
    offsets = linspace(-0.16, 0.16, nMethods);

    hold on;
    for mm = 1:nMethods
        data = iterCell{mm};
        for nn = 1:nCases
            y = data(:,nn);
            y = y(isfinite(y));
            if isempty(y)
                continue;
            end
            x = (centers(nn)+offsets(mm))*ones(size(y));
            boxchart(x, y, 'BoxFaceColor', colors(mm,:), ...
                'MarkerColor', colors(mm,:), 'BoxWidth', 0.22);
        end
    end

    legendHandles = gobjects(nMethods,1);
    for mm = 1:nMethods
        legendHandles(mm) = plot(nan, nan, 's', ...
            'MarkerFaceColor', colors(mm,:), ...
            'MarkerEdgeColor', colors(mm,:), ...
            'MarkerSize', 10, 'LineStyle', 'none');
    end

    xticks(centers);
    xticklabels(string(nTrainValues));
    xlim([0.5 nCases+0.5]);
    xlabel('$n_{\mathrm{train}}$', 'Interpreter', 'latex', 'FontSize', 27);
    ylabel('Maximum outer VB/EM iterations', ...
        'Interpreter', 'latex', 'FontSize', 25);
    set(gca, 'FontSize', 21, 'LineWidth', 1.3, ...
        'TickLabelInterpreter', 'latex');
    legend(legendHandles, methodNames, 'Location', 'northeast', ...
        'FontSize', 16, 'Interpreter', 'latex');
    grid off;
    box on;
    hold off;
    drawnow;
end


function save_scaling_figure(figHandle, figBase)
    set(figHandle, 'PaperPositionMode', 'auto');
    set(figHandle, 'Renderer', 'painters');
    set(figHandle, 'InvertHardcopy', 'off');

    try
        exportgraphics(figHandle, [figBase '.png'], 'Resolution', 300);
    catch ME
        warning('Could not save PNG %s: %s', figBase, ME.message);
    end

    try
        print(figHandle, [figBase '.eps'], '-depsc', '-painters');
    catch ME
        warning('Could not save EPS %s: %s', figBase, ME.message);
    end

    try
        savefig(figHandle, [figBase '.fig']);
    catch ME
        warning('Could not save FIG %s: %s', figBase, ME.message);
    end
end


function model = studentt_gpml_mogp_fit(X, Y, opts)
%STUDENTT_GPML_MOGP_FIT
% Fair independent-output Student-t GPR baseline using GPML.
%
% Fairness choices:
%   - no internal standardization of X
%   - no internal standardization of Y
%   - initialized from the same contaminated-data opts_run used by ASOR/GMM/RCGP
%   - likelihood sigma initialized from opts.init_R, not SigmaTrue
%   - kernel lengthscale initialized from opts.init_lengthscale
%   - signal std initialized from opts.init_signal_std
%   - mean initialized from contaminated median(Y)
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

    % Same fixed incorrect R initialization given to non-oracle methods.
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
        % This is NOT SigmaTrue. It comes from opts_run.init_R.
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
% No de-standardization is done because the model is trained directly
% on raw contaminated-output scale.

    nTest = size(Xtest,1);
    p = model.p;

    Ypred = zeros(nTest,p);

    % Ensure likelihood wrapper has same fixed nu used in training.
    set_student_gpml_fixed_nu(model.nu);

    for q = 1:p
        out = model.output{q};

        if out.ok
            % Reuse posterior struct and suppress output.
            evalc('[~, ~, fmu, ~] = gp(out.hyp, out.inffunc, out.meanfunc, out.covfunc, out.likfunc, out.X, out.y, Xtest);');

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
%STANDARD_MOGP_FIT Plain Gaussian scalar-output GP baseline.
%
% Runtime-fair implementation:
%   y = f + e, e ~ N(0, sigma0^2)
%
% Important:
%   - sigma0^2 is fixed.
%   - sigma0^2 is NOT SigmaTrue.
%   - sigma0^2 comes from opts.init_R, the shared wrong initialization.
%   - only kernel parameters and the constant mean are fitted.
%   - kernel parameters are optimized with analytic-gradient L-BFGS.
%   - no full posterior covariance or explicit K^{-1} is formed.

    [n,d] = size(X);
    p = size(Y,2);

    if p ~= 1
        error('Fair standard_mogp_fit is scalar-output only. Use independent_standard_gpr_fit for multi-output data.');
    end

    y = double(Y(:));

    if ~isfield(opts,'init_R')
        error('Standard GP requires opts.init_R. Do not initialize R from data.');
    end

    R0 = make_spd(opts.init_R, opts.jitter);
    noiseVar = R0(1,1);

    logtheta0 = [log(opts.init_lengthscale*ones(d,1)); ...
                 log(opts.init_signal_std)];

    [logtheta, stdInfo] = standard_fixedR_nlml_gradient_descent( ...
        logtheta0, X, y, noiseVar, opts);

    [bestVal, ~, aux] = standard_scalar_fixed_noise_nlml_grad( ...
        logtheta, X, y, noiseVar, opts.jitter);

    model = struct();
    model.X = X;
    model.Y = y;
    model.n = n;
    model.p = 1;
    model.m_hat = aux.m_hat;
    model.alpha_exact = aux.alpha;
    model.logtheta = logtheta;
    model.L = aux.L;
    model.noiseVar = noiseVar;
    model.nlml = bestVal;
    model.fitInfo = stdInfo;
    model.method = 'Standard-GPR-FairLean';

    % Keep these fields empty to avoid accidental full-covariance timing cost.
    model.mu_f = [];
    model.Sigma_f = [];
    model.Kinv = [];
end

function [Ypred, VarPred] = shared_mogp_predict(model, Xtest)
    Xtrain = model.X;
    n = model.n;
    p = model.p;
    nTest = size(Xtest,1);

    if isfield(model, 'Kinv')
        Kinv = model.Kinv;
    else
        Kinv = [];
    end

    if isfield(model, 'mu_f')
        mu_f = model.mu_f;
    else
        mu_f = [];
    end

    if isfield(model, 'Sigma_f')
        Sigma_f = model.Sigma_f;
    else
        Sigma_f = [];
    end

    m = model.m_hat;
    mf = kron(ones(n,1), m);
    logtheta = model.logtheta;
    ell = exp(logtheta(1:end-1));
    sf  = exp(logtheta(end));
    sf2 = sf^2;

    Ypred = zeros(nTest,p);
    VarPred = zeros(p,p,nTest);

    for t = 1:nTest
        xstar = Xtest(t,:);
        kstar = ard_rbf_kernel_cross_single(xstar, Xtrain, ell, sf2);
        KstarX = kron(kstar, eye(p));
        Kss = sf2 * eye(p);

        % Lean exact-GP path used by Standard GPR.  It stores alpha_exact
        % and the Cholesky factor L, but it intentionally does not store
        % the full posterior covariance or Kinv.
        if isfield(model, 'alpha_exact') && (isempty(Kinv) || isempty(Sigma_f))
            mu_star = m + KstarX * model.alpha_exact;

            if p == 1 && isfield(model, 'L') && ~isempty(model.L)
                v = model.L \ kstar';
                Sigma_star = max(sf2 - sum(v.^2), 1e-12);
            else
                Sigma_star = Kss;
            end

        else
            if isfield(model, 'alpha_exact')
                mu_star = m + KstarX * model.alpha_exact;
            else
                mu_star = m + KstarX*Kinv*(mu_f - mf);
            end

            Sigma_star = Kss - KstarX*Kinv*KstarX' + KstarX*Kinv*Sigma_f*Kinv*KstarX';
            Sigma_star = 0.5*(Sigma_star + Sigma_star');
        end

        Ypred(t,:) = mu_star';
        VarPred(:,:,t) = Sigma_star;
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
% Pluggable kernel optimizer for the ASOR/GMM variational kernel objective.
%
% Supported opts.thetaOptimizer:
%   'gd_linesearch'       : current normalized GD + Armijo backtracking
%   'gd_warmstart'        : reuses/increases the previous accepted step
%                           within this kernel-update call
%   'momentum_linesearch' : momentum direction + Armijo backtracking
%   'adam_linesearch'     : Adam-style adaptive direction + Armijo
%
% All modes keep the Armijo line search, so the kernel objective should not
% increase during accepted inner steps.

    if isfield(opts, 'thetaGradMaxIter')
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

    if isfield(opts, 'thetaGradTol')
        gradTol = opts.thetaGradTol;
    else
        gradTol = 1e-8;
    end

    if isfield(opts, 'thetaInnerRelTol')
        relTol = opts.thetaInnerRelTol;
    elseif isfield(opts, 'tol')
        relTol = opts.tol;
    else
        relTol = 1e-5;
    end

    if isfield(opts, 'thetaMinInnerIter')
        minInnerIter = opts.thetaMinInnerIter;
    else
        minInnerIter = 2;
    end

    if isfield(opts, 'thetaOptimizer')
        mode = lower(char(opts.thetaOptimizer));
    else
        mode = 'gd_linesearch';
    end

    validModes = {'gd_linesearch','gd_warmstart','momentum_linesearch','adam_linesearch'};
    if ~any(strcmp(mode, validModes))
        error('Unknown opts.thetaOptimizer = %s', mode);
    end

    if isfield(opts, 'thetaStepIncrease')
        stepIncrease = opts.thetaStepIncrease;
    else
        stepIncrease = 1.25;
    end

    if isfield(opts, 'thetaGradStepMinStart')
        minStart = opts.thetaGradStepMinStart;
    else
        minStart = minStep;
    end

    if isfield(opts, 'thetaGradStepMaxStart')
        maxStart = opts.thetaGradStepMaxStart;
    else
        maxStart = step0;
    end

    if isfield(opts, 'thetaMomentumBeta')
        betaMom = opts.thetaMomentumBeta;
    else
        betaMom = 0.90;
    end

    if isfield(opts, 'thetaAdamBeta1')
        beta1 = opts.thetaAdamBeta1;
    else
        beta1 = 0.90;
    end

    if isfield(opts, 'thetaAdamBeta2')
        beta2 = opts.thetaAdamBeta2;
    else
        beta2 = 0.999;
    end

    if isfield(opts, 'thetaAdamEps')
        adamEps = opts.thetaAdamEps;
    else
        adamEps = 1e-8;
    end

    logtheta = project_logtheta_bounds(logtheta0, X);

    [J, g] = kernel_variational_objective_grad( ...
        logtheta, X, mu_f, Sigma_f, m_hat, p, opts.jitter);

    info = struct();
    info.mode = mode;
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
    velocity = zeros(size(logtheta));
    adamM = zeros(size(logtheta));
    adamV = zeros(size(logtheta));
    adamT = 0;

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

        steepest = -g / max(1, gradNorm);

        switch mode

            case 'gd_linesearch'
                direction = steepest;
                trialStep = step0;

            case 'gd_warmstart'
                direction = steepest;
                trialStep = min(maxStart, max(minStart, stepStart));

            case 'momentum_linesearch'
                rawDirection = betaMom * velocity + steepest;

                if ~all(isfinite(rawDirection)) || g' * rawDirection >= 0
                    rawDirection = steepest;
                    velocity = zeros(size(velocity));
                end

                direction = rawDirection / max(1, norm(rawDirection));
                trialStep = min(maxStart, max(minStart, stepStart));

            case 'adam_linesearch'
                adamT = adamT + 1;
                adamM = beta1 * adamM + (1 - beta1) * g;
                adamV = beta2 * adamV + (1 - beta2) * (g.^2);

                mhat = adamM / max(1 - beta1^adamT, realmin);
                vhat = adamV / max(1 - beta2^adamT, realmin);

                rawDirection = -mhat ./ (sqrt(vhat) + adamEps);

                if ~all(isfinite(rawDirection)) || g' * rawDirection >= 0
                    rawDirection = steepest;
                end

                direction = rawDirection / max(1, norm(rawDirection));
                trialStep = min(maxStart, max(minStart, stepStart));
        end

        [accepted, cand, Jcand, gcand, acceptedStep, nBacktrack] = ...
            theta_armijo_line_search_local( ...
            logtheta, J, g, direction, X, mu_f, Sigma_f, m_hat, p, ...
            opts.jitter, trialStep, c1, minStep);

        if ~accepted
            info.lineSearchFailed = true;
            stepStart = max(minStart, 0.5 * trialStep);
            break;
        end

        Jprev = J;
        logtheta = cand;
        J = Jcand;
        g = gcand;

        relJChange = abs(Jprev - J) / max(1, abs(Jprev));

        info.nSteps = kk;
        info.lastStep = acceptedStep;
        info.Jfinal = J;
        info.gradNorm = norm(g);
        info.finalRelJChange = relJChange;

        stepTrace(kk) = acceptedStep;
        backtrackTrace(kk) = nBacktrack;

        if strcmp(mode, 'gd_linesearch')
            stepStart = step0;
        else
            stepStart = min(maxStart, max(minStart, stepIncrease * acceptedStep));
        end

        if strcmp(mode, 'momentum_linesearch')
            velocity = direction;
        end

        if kk >= minInnerIter && relJChange < relTol
            info.converged = true;
            info.convergedReason = 'relative_objective';
            break;
        end
    end

    if info.nSteps >= maxIter && ~info.converged
        info.hitMaxInner = true;
    end

    validBack = backtrackTrace(isfinite(backtrackTrace));
    validStep = stepTrace(isfinite(stepTrace));

    if ~isempty(validBack)
        info.meanBacktracks = mean(validBack);
    end

    if ~isempty(validStep)
        info.lastStep = validStep(end);
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
% This helper was required by the newer gd_warmstart / momentum / Adam kernel
% optimizer but was missing from the previous merged main simulation file.

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

function Y = true_multioutput_function(X, p, kind)
%TRUE_MULTIOUTPUT_FUNCTION Flexible synthetic multi-output function.
%
% Works for:
%   input dimension d = 1,2,3,...
%   output dimension p = 1,2,3,...
%
% X is n by d.
% Y is n by p.

    if nargin < 3
        kind = 'rbf_bumps';
    end

    [n,d] = size(X);
    Y = zeros(n,p);

    switch lower(kind)

        case 'rbf_bumps'
            % Smooth RBF-bump function generalized to d dimensions.
            % This is the best debugging function.

            centers = zeros(3,d);
            for k = 1:3
                centers(k,:) = -2 + (k-1) * (4/(3-1)) * ones(1,d);
            end

            width = 1.25 * sqrt(d);
            amps_base = [1.00, -0.80, 0.65];

            for q = 1:p
                yq = zeros(n,1);

                for k = 1:3
                    diff = X - centers(k,:);
                    r2 = sum(diff.^2, 2);
                    amp_qk = amps_base(k) * (1 + 0.10*(q-1));
                    yq = yq + amp_qk * exp(-0.5*r2/(width^2));
                end

                % Add mild output-specific smooth variation.
                yq = yq + 0.15*q*sin(0.5*X(:,1));

                Y(:,q) = yq;
            end

        case 'sinusoidal'
            % Smooth multidimensional sin/cos function.

            for q = 1:p
                yq = zeros(n,1);

                for k = 1:d
                    yq = yq ...
                        + (0.60/k) * sin((0.60 + 0.10*q + 0.05*k) * X(:,k)) ...
                        + (0.30/k) * cos((1.00 + 0.05*q + 0.03*k) * X(:,k));
                end

                if d >= 2
                    yq = yq + 0.20*sin(0.35*X(:,1).*X(:,2));
                end

                Y(:,q) = yq;
            end
                 case 'mixed_local_spikes'
            % Harder latent function:
            % smooth global trend + localized sharp structures.
            %
            % This is still smooth, but it has local features that are
            % harder for simple GP smoothing and heavy-tailed scalar models.

            Y = zeros(n,p);

            % Use first two coordinates for the main spatial structure.
            x1 = X(:,1);

            if d >= 2
                x2 = X(:,2);
            else
                x2 = zeros(n,1);
            end

            for q = 1:p

                % Smooth background component.
                yq = 0.60*sin(0.80*x1 + 0.15*q) ...
                   + 0.40*cos(0.70*x2 - 0.10*q);

                % Medium-scale nonlinear interaction.
                yq = yq + 0.35*sin(0.45*x1.*x2 + 0.05*q);

                % Localized positive bump.
                c1 = -1.5 + 0.15*q;
                c2 =  1.0 - 0.10*q;
                r2a = (x1 - c1).^2 + (x2 - c2).^2;
                yq = yq + 1.25*exp(-r2a / 0.18);

                % Localized negative bump.
                c3 =  1.4 - 0.10*q;
                c4 = -1.2 + 0.08*q;
                r2b = (x1 - c3).^2 + (x2 - c4).^2;
                yq = yq - 1.00*exp(-r2b / 0.12);

                % Add dependence on extra dimensions if d > 2.
                for k = 3:d
                    yq = yq + (0.20/k)*sin((0.7 + 0.05*q)*X(:,k));
                end

                Y(:,q) = yq;
            end

                        case 'friedman_like'
            % Friedman-style nonlinear regression function.
            % This is a classic benchmark-type latent function:
            % nonlinear interactions + quadratic terms + linear terms.
            %
            % Works for d_x >= 2, and safely handles d_x < 5.

            Y = zeros(n,p);

            x1 = X(:,1);

            if d >= 2
                x2 = X(:,2);
            else
                x2 = zeros(n,1);
            end

            if d >= 3
                x3 = X(:,3);
            else
                x3 = zeros(n,1);
            end

            if d >= 4
                x4 = X(:,4);
            else
                x4 = zeros(n,1);
            end

            if d >= 5
                x5 = X(:,5);
            else
                x5 = zeros(n,1);
            end

            % Map X roughly from [-3,3] to [0,1] for Friedman-style terms.
            z1 = (x1 + 3)/6;
            z2 = (x2 + 3)/6;
            z3 = (x3 + 3)/6;
            z4 = (x4 + 3)/6;
            z5 = (x5 + 3)/6;

            for q = 1:p

                yq = 10*sin(pi*z1.*z2) ...
                   + 20*(z3 - 0.5).^2 ...
                   + 10*z4 ...
                   + 5*z5;

                % Output-specific nonlinear variation.
                yq = yq ...
                   + 0.75*sin((0.8 + 0.1*q)*x1) ...
                   + 0.50*cos((0.6 + 0.05*q)*x2);

                % Mild output-specific shift before later normalization.
                yq = yq + 0.10*q*x1;

                Y(:,q) = yq;
            end

                        case 'nonstationary_multiscale'
            % Nonstationary multiscale latent function.
            %
            % Smooth globally, but with local high-frequency behavior.
            % This is harder than rbf_bumps and often exposes differences
            % between robust likelihoods and robust weighting methods.

            Y = zeros(n,p);

            x1 = X(:,1);

            if d >= 2
                x2 = X(:,2);
            else
                x2 = zeros(n,1);
            end

            % Local gate: activates high-frequency structure in one region.
            gate = exp(-((x1 - 0.75).^2 + (x2 + 0.50).^2) / 0.45);

            for q = 1:p

                % Global smooth trend.
                yq = 0.80*sin(0.65*x1 + 0.10*q) ...
                   + 0.60*cos(0.55*x2 - 0.08*q);

                % Nonlinear interaction.
                yq = yq + 0.35*sin(0.35*x1.*x2 + 0.05*q);

                % Local high-frequency component.
                yq = yq + gate .* ...
                    (0.85*sin((3.0 + 0.25*q)*x1) .* cos((2.5 + 0.15*q)*x2));

                % Local sharp bump.
                r2 = (x1 + 1.25 - 0.08*q).^2 + (x2 - 1.00 + 0.05*q).^2;
                yq = yq + 0.90*exp(-r2/0.10);

                % Extra dimensions, if present.
                for k = 3:d
                    yq = yq + (0.20/k)*sin((0.7 + 0.05*q)*X(:,k));
                end

                Y(:,q) = yq;
            end

        otherwise
            error('Unknown true function kind: %s', kind);
    end

    % Center each output.
    % Y = Y - mean(Y,1);

    % Add mild output correlation if p > 1.
    if p > 1
        C = eye(p) + 0.10*(ones(p) - eye(p));
        Y = Y * C;
    end
end

function X = mvnrnd_local(mu, Sigma, n)
    X = randn(n, numel(mu)) * chol(make_spd(Sigma, 1e-10), 'lower')' + repmat(mu(:)', n, 1);
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

function [Xtrain, Xtest, YcleanTrain, YcleanTest] = generate_synthetic_dataset( ...
    nTrain, nTest, d_x, p_y, dataMode, trueFunctionKind, mc)
%GENERATE_SYNTHETIC_DATASET Flexible synthetic dataset generator.
%
% dataMode:
%   'debug_grid' : structured grid-like input, easiest for debugging
%   'random_box' : random design in [-3,3]^d

    switch lower(dataMode)

        case 'debug_grid'
            if d_x == 1
                Xtrain = linspace(0,20,nTrain)';
                Xtest  = linspace(0,20,nTest)';
            else
                % For d_x > 1, use a quasi-grid-like random design.
                % This keeps the setup smoother than pure random testing.
                rng(2000 + mc);
                Xtrain = -3 + 6*lhsdesign_local(nTrain, d_x);
                Xtest  = -3 + 6*lhsdesign_local(nTest,  d_x);
            end

        case 'random_box'
            rng(2000 + mc);
            Xtrain = -3 + 6*rand(nTrain, d_x);
            Xtest  = -3 + 6*rand(nTest,  d_x);

        otherwise
            error('Unknown dataMode: %s', dataMode);
    end

    YcleanTrain = true_multioutput_function(Xtrain, p_y, trueFunctionKind);
    YcleanTest  = true_multioutput_function(Xtest,  p_y, trueFunctionKind);

    % trainMean = mean(YcleanTrain,1);
    % YcleanTrain = YcleanTrain - trainMean;
    % YcleanTest  = YcleanTest  - trainMean;

    trainMean = mean(YcleanTrain,1);
    trainStd  = std(YcleanTrain,0,1);
    
    trainStd(~isfinite(trainStd) | trainStd < 1e-8) = 1;
    
    YcleanTrain = (YcleanTrain - trainMean) ./ trainStd;
    YcleanTest  = (YcleanTest  - trainMean) ./ trainStd;
end

function X = lhsdesign_local(n,d)
%LHSDESIGN_LOCAL Simple Latin-hypercube-like design without toolbox dependency.

    X = zeros(n,d);

    for j = 1:d
        edges = linspace(0,1,n+1)';
        u = edges(1:n) + rand(n,1)/n;
        X(:,j) = u(randperm(n));
    end
end


function outlierNoise = generate_outlier_noise(n, p, outlierScale, outlierModel)
%GENERATE_OUTLIER_NOISE Flexible outlier noise generator.

    switch lower(outlierModel)

        case 'positive_shift'
            % Best for debugging monotonic degradation.
            outlierNoise = outlierScale * ones(n,p);

        case 'uniform'
            % Symmetric uniform outliers.
            outlierNoise = outlierScale * (2*rand(n,p) - 1);
        case 'positive_uniform'
            % Positive uniform outliers on [0, outlierScale].
            % Mean = outlierScale/2, variance = outlierScale^2/12.
            outlierNoise = outlierScale * rand(n,p);

        case 'negative_uniform'
            % Negative uniform outliers on [-outlierScale, 0].
            % Mean = -outlierScale/2, variance = outlierScale^2/12.
            outlierNoise = -outlierScale * rand(n,p);

        case 'gaussian'
            % Gaussian outliers with std = outlierScale.
            outlierNoise = outlierScale * randn(n,p);

        case 'alternating'
            % Deterministic signs, useful if you do not want all positive shifts.
            s = ones(n,p);
            s(2:2:end,:) = -1;
            outlierNoise = outlierScale * s;
         case 'partial_uniform'
            outlierNoise = zeros(n,p);
            for i = 1:n
                q = randi(p);  % corrupt only one random output dimension
                outlierNoise(i,q) = outlierScale * (2*rand - 1);
            end

        otherwise
            error('Unknown outlierModel: %s', outlierModel);
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
        % diagonal entry of the same fixed incorrect covariance.
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
        [yq_pred, ~] = shared_mogp_predict(model.output{q}, Xtest);
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
        [yq_pred, ~] = shared_mogp_predict(model.output{q}, Xtest);
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

function draw_grouped_filled_boxplots_boxchart_labels(dataCell, xLabels, methodNames, colors)
    nMethods = numel(dataCell);
    nX = numel(xLabels);
    nMC = size(dataCell{1},1);

    centers = 1:nX;
    delta = 0.12;
    offs = ((1:nMethods) - mean(1:nMethods)) * delta;

    hold on;
    bw = 0.11;
    h = gobjects(nMethods,1);
    allVals = [];

    for j = 1:nMethods
        Xpos = repelem(centers + offs(j), nMC).';
        Yvec = reshape(dataCell{j}, [], 1);
        allVals = [allVals; Yvec]; %#ok<AGROW>

        h(j) = boxchart(Xpos, Yvec, ...
            'BoxWidth', bw, ...
            'BoxFaceColor', colors(j,:), ...
            'BoxFaceAlpha', 0.60);

        h(j).LineWidth = 1.2;
    end

    for s = 1:(nX-1)
        xline(s + 0.5, ':', ...
            'Color', [0.45 0.45 0.45], ...
            'LineWidth', 1.4, ...
            'HandleVisibility', 'off');
    end

    yMax = 1.1 * max([allVals; 0.1], [], 'omitnan');

    xlim([0.5, nX+0.5]);
    ylim([0, yMax]);

    xticks(centers);
    xticklabels(xLabels);
    xtickangle(0);

    legend(h, methodNames, ...
        'Location', 'northwest', ...
        'AutoUpdate', 'off', ...
        'FontSize', 30);
end


function [logtheta, info] = standard_fixedR_nlml_gradient_descent(logtheta0, X, Y, R_fixed, opts)
%STANDARD_FIXEDR_NLML_GRADIENT_DESCENT
% Analytic-gradient L-BFGS optimizer for Standard GP fixed-noise NLML.
%
% The function name is kept for compatibility with older code, but this
% fair runtime version does not use numerical finite differences.

    y = Y(:);

    if isscalar(R_fixed)
        noiseVar = R_fixed;
    else
        R_fixed = make_spd(R_fixed, opts.jitter);
        noiseVar = R_fixed(1,1);
    end

    d = size(X,2);
    xRange = max(X,[],1) - min(X,[],1);
    xRange(~isfinite(xRange) | xRange <= 0) = 1;

    yScale = std(y);
    if ~isfinite(yScale) || yScale < 1e-8
        yScale = 1;
    end

    lower = [log(max(1e-3*xRange(:), 1e-5)); ...
             log(max(1e-5*yScale, 1e-6))];

    upper = [log(max(100*xRange(:), 1)); ...
             log(max(100*yScale, 10))];

    logtheta0 = min(max(logtheta0(:), lower), upper);

    objective = @(lt) standard_scalar_fixed_noise_nlml_grad( ...
        lt, X, y, noiseVar, opts.jitter);

    optBox = opts;
    if isfield(opts, 'fixedGPGradMaxIter')
        optBox.maxIter = opts.fixedGPGradMaxIter;
    end

    [logtheta, info] = rcgp_lbfgs_box(objective, logtheta0, lower, upper, optBox);
end

function [logtheta, info] = oracle_hetero_fixed_noise_nlml_gradient_descent( ...
    logtheta0, X, y, noiseVarVec, opts)
%ORACLE_HETERO_FIXED_NOISE_NLML_GRADIENT_DESCENT
% Analytic-gradient L-BFGS optimizer for Oracle GP with known
% heteroscedastic observation variances.

    y = y(:);
    noiseVarVec = noiseVarVec(:);

    d = size(X,2);
    xRange = max(X,[],1) - min(X,[],1);
    xRange(~isfinite(xRange) | xRange <= 0) = 1;

    yScale = std(y);
    if ~isfinite(yScale) || yScale < 1e-8
        yScale = 1;
    end

    lower = [log(max(1e-3*xRange(:), 1e-5)); ...
             log(max(1e-5*yScale, 1e-6))];

    upper = [log(max(100*xRange(:), 1)); ...
             log(max(100*yScale, 10))];

    logtheta0 = min(max(logtheta0(:), lower), upper);

    objective = @(lt) oracle_scalar_hetero_nlml_grad( ...
        lt, X, y, noiseVarVec, opts.jitter);

    optBox = opts;
    if isfield(opts, 'fixedGPGradMaxIter')
        optBox.maxIter = opts.fixedGPGradMaxIter;
    end

    [logtheta, info] = rcgp_lbfgs_box(objective, logtheta0, lower, upper, optBox);
end

function [nlml, grad, aux] = standard_scalar_fixed_noise_nlml_grad( ...
    logtheta, X, y, noiseVar, jitter)
%STANDARD_SCALAR_FIXED_NOISE_NLML_GRAD
% Exact scalar GP negative log marginal likelihood with fixed noise variance.
% The constant mean is profiled out by GLS. The analytic gradient uses the
% envelope theorem, so the derivative of the profiled mean is not needed.

    X = double(X);
    y = double(y(:));
    noiseVar = double(noiseVar);

    [n,d] = size(X);

    aux = struct('m_hat', NaN, 'alpha', [], 'L', []);

    if noiseVar <= 0 || ~isfinite(noiseVar)
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    [Kbase, dKcell, bad] = se_ard_kernel_base_and_derivatives(logtheta, X);
    if bad
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    Ky = make_spd(Kbase + (noiseVar + jitter)*eye(n), jitter);

    [L,flag] = chol(Ky,'lower');
    if flag ~= 0
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    one = ones(n,1);

    KyInvY = L' \ (L \ y);
    KyInvOne = L' \ (L \ one);

    denom = one' * KyInvOne;
    if denom <= realmin || ~isfinite(denom)
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    m_hat = (one' * KyInvY) / denom;
    r = y - m_hat*one;

    alpha = L' \ (L \ r);
    Q = L' \ (L \ eye(n));
    Q = 0.5*(Q + Q');

    nlml = 0.5*(r' * alpha) ...
        + sum(log(diag(L))) ...
        + 0.5*n*log(2*pi);

    if ~isfinite(nlml)
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    G = 0.5 * (Q - alpha*alpha');
    G = 0.5*(G + G');

    grad = zeros(numel(logtheta),1);
    for k = 1:numel(logtheta)
        grad(k) = sum(sum(G .* dKcell{k}));
    end

    if any(~isfinite(grad))
        grad = zeros(size(logtheta));
    end

    aux.m_hat = m_hat;
    aux.alpha = alpha;
    aux.L = L;
end

function [nlml, grad, aux] = oracle_scalar_hetero_nlml_grad( ...
    logtheta, X, y, noiseVarVec, jitter)
%ORACLE_SCALAR_HETERO_NLML_GRAD
% Exact scalar GP negative log marginal likelihood with known heteroscedastic
% noise variances and analytic kernel gradient.

    X = double(X);
    y = double(y(:));
    noiseVarVec = double(noiseVarVec(:));

    [n,d] = size(X);

    aux = struct('m_hat', NaN, 'alpha', [], 'L', []);

    if numel(y) ~= n || numel(noiseVarVec) ~= n || ...
            any(noiseVarVec <= 0) || any(~isfinite(noiseVarVec))
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    [Kbase, dKcell, bad] = se_ard_kernel_base_and_derivatives(logtheta, X);
    if bad
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    Ky = make_spd(Kbase + diag(noiseVarVec) + jitter*eye(n), jitter);

    [L,flag] = chol(Ky,'lower');
    if flag ~= 0
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    one = ones(n,1);

    KyInvY = L' \ (L \ y);
    KyInvOne = L' \ (L \ one);

    denom = one' * KyInvOne;
    if denom <= realmin || ~isfinite(denom)
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    m_hat = (one' * KyInvY) / denom;
    r = y - m_hat*one;

    alpha = L' \ (L \ r);
    Q = L' \ (L \ eye(n));
    Q = 0.5*(Q + Q');

    nlml = 0.5*(r' * alpha) ...
        + sum(log(diag(L))) ...
        + 0.5*n*log(2*pi);

    if ~isfinite(nlml)
        nlml = 1e20;
        grad = zeros(size(logtheta));
        return;
    end

    G = 0.5 * (Q - alpha*alpha');
    G = 0.5*(G + G');

    grad = zeros(numel(logtheta),1);
    for k = 1:numel(logtheta)
        grad(k) = sum(sum(G .* dKcell{k}));
    end

    if any(~isfinite(grad))
        grad = zeros(size(logtheta));
    end

    aux.m_hat = m_hat;
    aux.alpha = alpha;
    aux.L = L;
end

function [Kbase, dKcell, bad] = se_ard_kernel_base_and_derivatives(logtheta, X)
%SE_ARD_KERNEL_BASE_AND_DERIVATIVES
% Base SE-ARD kernel and derivatives with respect to log lengthscales and
% log signal standard deviation. No jitter is included in Kbase.

    X = double(X);
    [n,d] = size(X);

    bad = false;
    dKcell = cell(d+1,1);

    ell = exp(logtheta(1:d));
    sf  = exp(logtheta(end));
    sf2 = sf^2;

    xRange = max(X,[],1) - min(X,[],1);
    xRange(xRange <= 0 | ~isfinite(xRange)) = 1;

    if any(ell < 1e-4*xRange(:)) || any(ell > 1e3*xRange(:)) || ...
            sf < 1e-8 || sf > 1e8 || any(~isfinite(ell)) || ~isfinite(sf)
        bad = true;
        Kbase = [];
        return;
    end

    Xscaled = X ./ ell(:)';
    D2 = squared_distance_matrix(Xscaled, Xscaled);

    Kbase = sf2 * exp(-0.5*D2);

    for q = 1:d
        xq = X(:,q);
        Dq2 = squared_distance_matrix(xq./ell(q), xq./ell(q));
        dKcell{q} = Kbase .* Dq2;
    end

    dKcell{end} = 2*Kbase;
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
        % wrong initial variance given to the other non-oracle methods.
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
        [mu_q, ~, ~] = rcgp_scalar_faithful_predict( ...
            model.output{q}, Xtest);

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


function [muStar, varLatentStar, varObservedStar] = ...
    rcgp_scalar_faithful_predict(model, Xstar)
%RCGP_SCALAR_FAITHFUL_PREDICT
% Closed-form paper-style RCGP posterior prediction.

    Xstar = double(Xstar);

    KtrainStar = rcgp_se_ard_kernel( ...
        model.X, Xstar, model.lengthscale, model.signalStd);

    muStar = model.meanConstant + KtrainStar' * model.alpha;

    V = model.L \ KtrainStar;

    varLatentStar = model.signalVar - sum(V.^2,1)';
    varLatentStar = max(varLatentStar, 1e-12);

    varObservedStar = varLatentStar + model.noiseVar;
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
%
% This optimizer is shared by RCGPR, Standard GPR, and Oracle.  The
% RCGPR call path already passes fields such as gradTol/stepTol/armijo,
% but the Standard/Oracle fixed-noise call paths pass the main opts
% structure.  Therefore we fill safe defaults here so the optimizer does
% not fail with missing fields such as opts.gradTol.

    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    if ~isfield(opts, 'maxIter') || isempty(opts.maxIter)
        opts.maxIter = 100;
    end

    if ~isfield(opts, 'lbfgsMemory') || isempty(opts.lbfgsMemory)
        if isfield(opts, 'rcgp_lbfgsMemory') && ~isempty(opts.rcgp_lbfgsMemory)
            opts.lbfgsMemory = opts.rcgp_lbfgsMemory;
        else
            opts.lbfgsMemory = 10;
        end
    end

    if ~isfield(opts, 'gradTol') || isempty(opts.gradTol)
        if isfield(opts, 'rcgp_gradTol') && ~isempty(opts.rcgp_gradTol)
            opts.gradTol = opts.rcgp_gradTol;
        elseif isfield(opts, 'thetaGradTol') && ~isempty(opts.thetaGradTol)
            opts.gradTol = opts.thetaGradTol;
        else
            opts.gradTol = 1e-5;
        end
    end

    if ~isfield(opts, 'stepTol') || isempty(opts.stepTol)
        if isfield(opts, 'rcgp_stepTol') && ~isempty(opts.rcgp_stepTol)
            opts.stepTol = opts.rcgp_stepTol;
        else
            opts.stepTol = 1e-10;
        end
    end

    if ~isfield(opts, 'armijo') || isempty(opts.armijo)
        if isfield(opts, 'rcgp_armijo') && ~isempty(opts.rcgp_armijo)
            opts.armijo = opts.rcgp_armijo;
        elseif isfield(opts, 'thetaGradArmijo') && ~isempty(opts.thetaGradArmijo)
            opts.armijo = opts.thetaGradArmijo;
        else
            opts.armijo = 1e-4;
        end
    end

    if ~isfield(opts, 'minStep') || isempty(opts.minStep)
        if isfield(opts, 'rcgp_minStep') && ~isempty(opts.rcgp_minStep)
            opts.minStep = opts.rcgp_minStep;
        elseif isfield(opts, 'thetaGradMinStep') && ~isempty(opts.thetaGradMinStep)
            opts.minStep = opts.thetaGradMinStep;
        else
            opts.minStep = 1e-10;
        end
    end

    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = false;
    end

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
        [yq_pred, ~] = shared_mogp_predict(model.output{q}, Xtest);
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
        [yq_pred, ~] = shared_mogp_predict(model.output{q}, Xtest);
        Ypred(:,q) = yq_pred(:,1);
    end
end


function model = asor_mogp_fit_grad_lrdiag(X, Y, opts)
%ASOR_MOGP_FIT_GRAD_LRDIAG
% Current scalar ASOR-GPR, with extra diagnostics for LR tuning.

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

        if ~isfield(opts, 'asor_R_update')
            opts.asor_R_update = 'iw_mode_original';
        end

        switch lower(opts.asor_R_update)

            case 'iw_mode_original'
                denom_R = n + opts.nu0 + p + 1;
                Sigma_hat = (S + S0) / max(denom_R, realmin);

            case 'weighted_denominator'
                denom_R = sum(w) + opts.nu0 + p + 1;
                Sigma_hat = (S + S0) / max(denom_R, realmin);

            case 'scalar_ig_conjugate_mean'
                if p ~= 1
                    error('scalar_ig_conjugate_mean is only valid for p = 1.');
                end
                alpha_N_sigma = opts.alpha0_sigma + 0.5*n;
                beta_N_sigma  = opts.beta0_sigma  + 0.5*S(1,1);
                sigma2_hat = beta_N_sigma / max(alpha_N_sigma - 1, realmin);
                Sigma_hat = sigma2_hat;

            case 'scalar_ig_conjugate_map'
                if p ~= 1
                    error('scalar_ig_conjugate_map is only valid for p = 1.');
                end
                alpha_N_sigma = opts.alpha0_sigma + 0.5*n;
                beta_N_sigma  = opts.beta0_sigma  + 0.5*S(1,1);
                sigma2_hat = beta_N_sigma / max(alpha_N_sigma + 1, realmin);
                Sigma_hat = sigma2_hat;

            otherwise
                error('Unknown opts.asor_R_update = %s', opts.asor_R_update);
        end

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
        'asor_R_update', opts.asor_R_update, ...
        'fitInfo', fitInfo);
end


function model = gmm_mogp_fit_lrdiag(X, Y, opts)
%GMM_MOGP_FIT_LRDIAG
% Current scalar GMM-GPR, with extra diagnostics for LR tuning.

    [n, d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    m_hat = median(Y,1)';

    if ~isfield(opts, 'init_R')
        error('GMM-GPR-LRDiag requires opts.init_R.');
    end

    Sigma_base = make_spd(opts.init_R, opts.jitter);

    if isfield(opts, 'gmm_outlier_init_multiplier')
        gmmOutlierInitMultiplier = opts.gmm_outlier_init_multiplier;
    else
        gmmOutlierInitMultiplier = 10.0;
    end

    if isfield(opts, 'gmm_nominal_init_multiplier')
        gmmNominalInitMultiplier = opts.gmm_nominal_init_multiplier;
    else
        gmmNominalInitMultiplier = 1.0;
    end

    % Component covariance initialization:
    %   component 1 = broad/outlier component
    %   component 2 = narrow/nominal component
    %
    % This uses only the shared non-oracle init_R and fixed multipliers.
    % It does not use p_out, true labels, or any oracle variance.
    Sigma_out0 = make_spd(gmmOutlierInitMultiplier * Sigma_base, opts.jitter);
    Sigma_nom0 = make_spd(gmmNominalInitMultiplier * Sigma_base, opts.jitter);

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

            % Paper-faithful EM covariance update.
            % No covariance prior or oracle regularization is used.
            Sigma_comp{j} = make_spd(S_j / max(sum_gamma, realmin), opts.jitter);
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
