%% Real_Data_AirQuality_Uneg5_15_Only_With_Oracle_PARALLEL_MC_TEST
% Automated real-data robustness experiment with Oracle added.
% Updated to use the current ASOR/GMM LR-diagnostic independent-output setup.
%
% Runs exactly one isolated real-data case for debugging:
%   1. AirQuality: U[-5,15]
%
% For each case, sweeps:
%   p_out = 0, 0.1, ..., 0.8
%
% Methods:
%   1. Oracle       -- perfect entry-wise outlier labels and best-guess
%                     real-data inlier variance
%   2. ASOR-GPR     -- new LR-diagnostic scalar-output setup
%   3. GMM-GPR      -- new LR-diagnostic scalar-output setup
%   4. Student-t GPR
%   5. RCGPR        -- faithful scalar paper-style implementation
%   6. Standard GPR
%
% Real-data protocol:
%   1. Load clean real train/test split.
%   2. Standardize X and Y using training statistics only.
%   3. Treat clean standardized Ytrain/Ytest as the reference.
%   4. Add artificial outliers only to Ytrain.
%   5. Evaluate RMSE against clean Ytest.
%
% Oracle protocol for real data:
%   - Oracle trains on the same contaminated Ytrain as all other methods.
%   - Oracle receives the true artificially injected entry-wise outlier mask.
%   - Oracle receives a best-guess nominal inlier variance because the true
%     real-data measurement variance is not known.
%   - Oracle receives a broad outlier variance equal to
%       oracle_inlier_var + E[outlier_noise^2].
%
clear; clc; close all;
rng(7);

%% ========================================================================
% GPML setup for Student-t GP baseline
%% ========================================================================
gpml_path = 'C:\Users\majal\Downloads\gpml-matlab-v4.2-2018-06-11';
addpath(genpath(gpml_path));

% Suppress GPML startup printing.
evalc('startup;');

%% ========================================================================
% Parallel setup for Monte Carlo runs
% ========================================================================
% This script parallelizes the independent Monte Carlo runs using parfor.
% Start conservatively with 4 workers. Test 6 or 8 later if the machine has
% enough CPU cores and memory. More workers is not always faster because
% each worker performs many dense linear-algebra solves.
useParallelMC = true;  % Leave true for this parallel-testing script.
numWorkers = 8;        % Try 4 first, then test 6 or 8.

if license('test','Distrib_Computing_Toolbox') ~= 1
    error('Parallel Computing Toolbox is required for this parallel script.');
end

pool = gcp('nocreate');

if isempty(pool)
    pool = parpool('local', numWorkers);
else
    fprintf('\nUsing existing parallel pool with %d workers.\n', pool.NumWorkers);
end

% Make GPML visible on every worker.
f1 = parfevalOnAll(@addpath, 0, genpath(gpml_path));
wait(f1);

% Run GPML startup silently on every worker.
f2 = parfevalOnAll(@() evalc('startup;'), 0);
wait(f2);

% Avoid severe CPU oversubscription: each worker uses one BLAS thread.
% You can comment this out if testing shows it is slower on your machine.
f3 = parfevalOnAll(@maxNumCompThreads, 0, 1);
wait(f3);

fprintf('\nParallel MC enabled with %d workers.\n', pool.NumWorkers);

fprintf('\n============================================================\n');
fprintf('Isolated AirQuality U[-5,15] Real-Data Comparison WITH Oracle: Oracle, ASOR-GPR, GMM-GPR, Student-t GPR, RCGPR, Standard GPR\n');
fprintf('p_out sweep = 0:0.1:0.8 | Isolated case: AirQuality U[-5,15]\n');
fprintf('============================================================\n');

%% ========================================================================
% Global experiment settings
%% ========================================================================

nTrain = 100;
nTest  = 100;

% Use 30 for the paper figure. Use 5 only for quick debugging.
nMC = 30;

pOutList = 0:0.1:0.8;
nPout = numel(pOutList);

% Real-data initialization scale.
% We do not know the true nominal noise for real data, so all non-oracle
% methods receive the same common initialization.
sigma_init = 0.5;
sigma_init_wrong = sigma_init;
sigma_asor_prior = sigma_init;

% Oracle-only best guess for the unknown real-data nominal/inlier sigma.
% The real datasets are standardized and no extra nominal Gaussian noise is
% added, so this is intentionally better than the shared non-oracle
% initialization sigma_init = 0.5. Increase to 0.10 if the Oracle appears
% numerically too sharp for a particular real-data split.
oracleBestGuessSigma = 0.5;
oracleBestGuessInlierVar = oracleBestGuessSigma^2;

%% ========================================================================
% Results folder
% ========================================================================
% Use a short absolute path to avoid MATLAB EPS/print path-length errors.

outputRoot = 'C:\Users\majal';
outputDir  = fullfile(outputRoot, 'GPR_Real_Data_AirQuality_Uneg5_15_ONLY_GMM_DEBUG_PARALLEL_MC_TEST');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('\nResults will be saved in:\n  %s\n', outputDir);

%% ========================================================================
% Dataset/outlier case list: isolated AirQuality uniform case
%% ========================================================================

experiments = struct([]);

% --------------------------
% 1) AirQuality: U[-5,15]
% --------------------------
experiments(1).realDataName = 'airquality';
experiments(1).realDataPath = 'C:\Users\majal\Downloads\GPR_Datasets\GPR_Datasets\AirQualityUCI_clean_numeric.csv';
experiments(1).realOutputCols = [];
experiments(1).numOutputs = 3;
experiments(1).outlierModel = 'shifted_uniform';
experiments(1).outlierScale = [-5, 15];   % U[-5,15]
experiments(1).caseName = 'AirQuality_U_neg5_15';
experiments(1).caseLabel = 'Air Quality: $U(-5,15)$';

% Other dataset cases removed for this isolated AirQuality debugging script.

nExperiments = numel(experiments);

%% ========================================================================
% Select which real-data cases to run
%% ========================================================================
% Case numbers:
%   1 = AirQuality_U_neg5_15
%
% Run only the isolated AirQuality U[-5,15] case.
activeExperimentIdx = 1;

%% ========================================================================
% Shared model settings
%% ========================================================================

opts = struct();

% ------------------------------------------------------------------------
% UPDATED ASOR/GMM convergence and kernel-GD setup
% ------------------------------------------------------------------------
% These settings are taken from the current LR-diagnostic setup.
% They are applied to the ASOR-GPR and GMM-GPR outer VB/EM loops.

opts.maxIter = 1000;
opts.tol = 1e-5;
opts.outerConvergenceCriterion = 'obj_window3_max';

opts.init_signal_std = 1.0;
opts.init_lengthscale = 1.0;

% ARD kernel hyperparameter learning.
opts.learnTheta = true;

% Update ASOR/GMM kernel hyperparameters every VB/EM iteration.
opts.thetaBurnIn = 0;
opts.thetaUpdateEvery = 1;

% Warm-started analytic-gradient Armijo gradient descent.
opts.thetaOptimizer = 'gd_warmstart';
opts.thetaGradStep = 0.05;
opts.thetaGradMaxIter = 25;
opts.thetaMaxIter = opts.thetaGradMaxIter;
opts.thetaMaxFunEvals = 20;
opts.thetaGradArmijo = 1e-4;
opts.thetaGradMinStep = 1e-7;
opts.thetaGradVerbose = false;

% Decouple inner kernel tolerance from the outer practical stopping rule.
opts.thetaInnerRelTol = 1e-5;
opts.thetaGradTol = 1e-5;
opts.thetaMinInnerIter = 2;

% Warm-start / momentum / Adam-compatible options used by the current helper.
opts.thetaStepIncrease = 1.25;
opts.thetaGradStepMinStart = 1e-5;
opts.thetaGradStepMaxStart = opts.thetaGradStep;
opts.thetaMomentumBeta = 0.90;
opts.thetaAdamBeta1 = 0.90;
opts.thetaAdamBeta2 = 0.999;
opts.thetaAdamEps = 1e-8;

% One-shot optimization budget for Standard GPR.
% Standard GPR does not have an ASOR/GMM outer VB/EM loop.
opts.fixedGPGradMaxIter = 60;

% ASOR priors.
opts.a = 1;
opts.theta0 = 0.5;
opts.A = 10;
opts.nu0_base_offset = 2;

% Original independent ASOR-GPR IW-style prior scale.
nu0_scalar = 1 + opts.nu0_base_offset;
p_scalar   = 1;
opts.S0_scale = (nu0_scalar + p_scalar + 1) * sigma_asor_prior^2;

% Outlier branch broadness initialization for ASOR and GMM prior options.
robustScaleFactor = 20;
opts.init_b = opts.a * robustScaleFactor;
opts.B = (opts.A - 1) / opts.init_b;

% GMM covariance priors. Keep false to match the current setup.
opts.gmm_use_cov_prior = false;
opts.gmm_prior_strength_nom = 5.0;
opts.gmm_prior_strength_out = 5.0;
opts.gmm_nominal_prior_var = sigma_asor_prior^2;
opts.gmm_outlier_prior_var = sigma_asor_prior^2 * (opts.init_b / opts.a);

% Compatibility fields used by the current ASOR/GMM code path.
opts.alpha0_sigma = 1.2;
opts.beta0_sigma  = (opts.alpha0_sigma - 1) * sigma_init_wrong^2;

%% ------------------------------------------------------------
% RCGP parameters: faithful scalar paper-style RCGP
%
% Base paper-style epsilon. In the p_out loop below, opts_run.rcgp_epsilon
% is set to max(0.05, pOutNow), so p_out=0 still uses the paper default.
%% ------------------------------------------------------------
opts.rcgp_epsilon = 0.05;
opts.rcgp_useShrinkageTerm = true;
opts.rcgp_shrinkageConvention = 'section3';

% L-BFGS settings for faithful RCGP.
opts.rcgp_maxIter = 100;
opts.rcgp_lbfgsMemory = 10;
opts.rcgp_gradTol = 1e-5;
opts.rcgp_stepTol = 1e-10;
opts.rcgp_armijo = 1e-4;
opts.rcgp_minStep = 1e-10;
opts.rcgp_verbose = false;

%% ------------------------------------------------------------
% Student-t GPML settings
%% ------------------------------------------------------------
opts.student_gpml_inference = 'VB';
opts.student_gpml_nu = 4;
opts.student_gpml_optIters = -60;

opts.jitter = 1e-6;
opts.minWeight = 1e-6;
opts.verbose = false;

fprintf('\nUpdated ASOR/GMM convergence setup used in this real-data p_out sweep:\n');

if isfield(opts, 'thetaOptimizer')
    fprintf('  thetaOptimizer               = %s\n', opts.thetaOptimizer);
end

fprintf('  thetaGradStep                = %.4g\n', opts.thetaGradStep);
fprintf('  thetaGradMaxIter             = %d\n', opts.thetaGradMaxIter);
fprintf('  thetaInnerRelTol             = %.1e\n', opts.thetaInnerRelTol);
fprintf('  thetaGradTol                 = %.1e\n', opts.thetaGradTol);

if isfield(opts, 'outerConvergenceCriterion')
    fprintf('  outerConvergenceCriterion    = %s\n', opts.outerConvergenceCriterion);
end

fprintf('  outer tol                    = %.1e\n', opts.tol);
fprintf('  outer maxIter                = %d\n', opts.maxIter);

if isfield(opts, 'fixedGPGradMaxIter')
    fprintf('  fixedGPGradMaxIter           = %d\n', opts.fixedGPGradMaxIter);
end

%% ========================================================================
% Method list and colors
%% ========================================================================

methodNames = {'Oracle', 'ASOR-GPR', 'GMM-GPR', ...
               'Student-t GPR', 'RCGPR', 'Standard GPR'};

colors = [
    0.25 0.25 0.25;             % Oracle
    35 139 69;                  % ASOR-GPR
    180 60 120;                 % GMM-GPR
    0.75*255 0.10*255 0.10*255; % Student-t GPR
    20 120 180;                 % RCGPR
    220 120 20                  % Standard GPR
] / 255;

pOutLabels = strings(1,nPout);
for pp = 1:nPout
    pOutLabels(pp) = sprintf('%.1f', pOutList(pp));
end

allResults = struct([]);

%% ========================================================================
% Runtime timers
%% ========================================================================
totalWallTimer = tic;

%% ========================================================================
% Run the isolated AirQuality real-data case
%% ========================================================================

for ee = activeExperimentIdx

    expCfg = experiments(ee);

    [outlierVar, outlierSecondMoment, outlierMean] = outlier_stats_local( ...
        expCfg.outlierModel, expCfg.outlierScale);

    % Oracle uses the true artificial outlier labels and a best-guess
    % real-data nominal variance.  The broad Oracle outlier variance uses
    % the second moment of the injected artificial outlier amplitude.
    oracleInlierObsVar  = oracleBestGuessInlierVar;
    oracleOutlierObsVar = oracleInlierObsVar + outlierSecondMoment;

    fprintf('\n\n============================================================\n');
    fprintf('Real-data case %d/%d: %s\n', ee, nExperiments, expCfg.caseName);
    fprintf('Dataset        = %s\n', expCfg.realDataName);
    fprintf('Path           = %s\n', expCfg.realDataPath);
    fprintf('nTrain/nTest   = %d/%d\n', nTrain, nTest);
    fprintf('nMC            = %d\n', nMC);
    fprintf('p_out values   = %s\n', mat2str(pOutList));
    fprintf('outlierModel   = %s\n', expCfg.outlierModel);
    fprintf('outlierScale   = %s\n', mat2str(expCfg.outlierScale));
    fprintf('outlierVar     = %.4f\n', outlierVar);
    fprintf('E[outlier]     = %.4f\n', outlierMean);
    fprintf('E[outlier^2]   = %.4f\n', outlierSecondMoment);
    fprintf('sigma_init, non-oracle      = %.4f\n', sigma_init);
    fprintf('Oracle best-guess sigma     = %.4f\n', oracleBestGuessSigma);
    fprintf('Oracle inlier variance      = %.6f\n', oracleInlierObsVar);
    fprintf('Oracle outlier variance     = %.6f\n', oracleOutlierObsVar);
    fprintf('============================================================\n');

    RMSE_ORACLE  = zeros(nMC,nPout);
    RMSE_ASOR    = zeros(nMC,nPout);
    RMSE_GMM     = zeros(nMC,nPout);
    RMSE_STUDENT = zeros(nMC,nPout);
    RMSE_RCGP    = zeros(nMC,nPout);
    RMSE_STD     = zeros(nMC,nPout);

    % Updated ASOR/GMM convergence and timing diagnostics.
    fitTime_ASOR = nan(nMC,nPout);
    fitTime_GMM  = nan(nMC,nPout);

    ASOR_outerConverged = false(nMC,nPout);
    GMM_outerConverged  = false(nMC,nPout);
    ASOR_hitMaxIter = false(nMC,nPout);
    GMM_hitMaxIter  = false(nMC,nPout);
    ASOR_iterMedian = nan(nMC,nPout);
    GMM_iterMedian  = nan(nMC,nPout);
    ASOR_iterMax = nan(nMC,nPout);
    GMM_iterMax  = nan(nMC,nPout);
    ASOR_finalRelMax = nan(nMC,nPout);
    GMM_finalRelMax  = nan(nMC,nPout);
    ASOR_innerConvFrac = nan(nMC,nPout);
    GMM_innerConvFrac  = nan(nMC,nPout);
    ASOR_innerHitCapFrac = nan(nMC,nPout);
    GMM_innerHitCapFrac  = nan(nMC,nPout);
    ASOR_lineSearchFailFrac = nan(nMC,nPout);
    GMM_lineSearchFailFrac  = nan(nMC,nPout);

    outlierEntryCount = zeros(nMC,nPout);
    outlierRowCount   = zeros(nMC,nPout);
    d_x_store = zeros(nMC,nPout);
    p_y_store = zeros(nMC,nPout);

    for pp = 1:nPout

        pOutNow = pOutList(pp);

        fprintf('\n============================================================\n');
        fprintf('%s | p_out case %d/%d: p_out = %.2f\n', ...
            expCfg.caseName, pp, nPout, pOutNow);
        fprintf('============================================================\n');

        ppWallTimer = tic;

        parfor mc = 1:nMC

            fprintf('\n------------------------------------------------------------\n');
            fprintf('%s | p_out %.2f | MC run %d/%d\n', expCfg.caseName, pOutNow, mc, nMC);
            fprintf('------------------------------------------------------------\n');

            seedNow = 100000 + 10000*ee + 100*pp + mc;
            rng(seedNow);

            [Xtrain, Xtest, YcleanTrain, YcleanTest] = load_real_numeric_split_csv( ...
                expCfg.realDataPath, nTrain, nTest, seedNow, ...
                expCfg.realOutputCols, expCfg.numOutputs);

            d_x = size(Xtrain, 2);
            p_y = size(YcleanTrain, 2);

            d_x_store(mc,pp) = d_x;
            p_y_store(mc,pp) = p_y;

            % For real data, the observed clean standardized dataset is treated
            % as the clean reference. No extra nominal Gaussian noise is added.
            Ynominal = YcleanTrain;

            % Common initialization covariance after p_y is known.
            SigmaInit = sigma_init^2 * eye(p_y);

            opts_run = opts;
            opts_run.nu0 = p_y + opts.nu0_base_offset;

            % All non-oracle methods receive the same fixed initialization.
            opts_run.init_R = SigmaInit;
            opts_run.sigma_init = sigma_init;
            opts_run.sigma_init_wrong = sigma_init_wrong;

            % Fixed RCGPR parameter for every contamination case.
            % It is not determined from the true p_out.
            opts_run.rcgp_epsilon = opts.rcgp_epsilon;

            % Shared kernel initialization from input design only.
            ell0 = 0.20 * median_pairwise_distance(Xtrain);
            if ~isfinite(ell0) || ell0 <= 0
                ell0 = opts.init_lengthscale;
            end
            ell0 = max(ell0, 1e-2);

            opts_run.init_lengthscale = ell0;
            opts_run.init_signal_std  = 1.0;
            opts_run.use_median_mean_init = false;

            %% ------------------------------------------------------------
            % Add entry-wise independent contamination to Ytrain only
            %% ------------------------------------------------------------
            Ytrain = Ynominal;

            outlierNoiseAll = generate_outlier_noise( ...
                nTrain, p_y, expCfg.outlierScale, expCfg.outlierModel);

            isOutlierEntry = rand(nTrain, p_y) < pOutNow;
            Ytrain(isOutlierEntry) = Ytrain(isOutlierEntry) + outlierNoiseAll(isOutlierEntry);

            nOutEntries = sum(isOutlierEntry(:));
            nTotalEntries = numel(isOutlierEntry);
            nOutRows = sum(any(isOutlierEntry, 2));

            outlierEntryCount(mc,pp) = nOutEntries;
            outlierRowCount(mc,pp) = nOutRows;

            %% ------------------------------------------------------------
            % 1. ASOR-GPR: current LR-diagnostic independent-output setup
            %% ------------------------------------------------------------
            tStartASOR = tic;
            model_asor = independent_asor_gpr_fit_lrdiag(Xtrain, Ytrain, opts_run);
            fitTime_ASOR(mc,pp) = toc(tStartASOR);

            Ypred_asor = independent_asor_gpr_predict_lrdiag(model_asor, Xtest);
            RMSE_ASOR(mc,pp) = sqrt(mean((Ypred_asor(:) - YcleanTest(:)).^2));

            diag_asor = collect_independent_fit_diagnostics_cap(model_asor);
            ASOR_outerConverged(mc,pp) = diag_asor.allConverged;
            ASOR_hitMaxIter(mc,pp) = diag_asor.anyHitMaxIter;
            ASOR_iterMedian(mc,pp) = diag_asor.iterMedian;
            ASOR_iterMax(mc,pp) = diag_asor.iterMax;
            ASOR_finalRelMax(mc,pp) = diag_asor.finalRelMax;
            ASOR_innerConvFrac(mc,pp) = diag_asor.thetaInnerConvFrac;
            ASOR_innerHitCapFrac(mc,pp) = diag_asor.thetaInnerHitCapFrac;
            ASOR_lineSearchFailFrac(mc,pp) = diag_asor.thetaLineSearchFailFrac;

            %% ------------------------------------------------------------
            % 2. GMM-GPR: current LR-diagnostic independent-output setup
            %% ------------------------------------------------------------
            tStartGMM = tic;
            model_gmm = independent_gmm_gpr_fit_lrdiag(Xtrain, Ytrain, opts_run);
            fitTime_GMM(mc,pp) = toc(tStartGMM);

            Ypred_gmm = independent_gmm_gpr_predict_lrdiag(model_gmm, Xtest);
            RMSE_GMM(mc,pp) = sqrt(mean((Ypred_gmm(:) - YcleanTest(:)).^2));

            diag_gmm = collect_independent_fit_diagnostics_cap(model_gmm);
            GMM_outerConverged(mc,pp) = diag_gmm.allConverged;
            GMM_hitMaxIter(mc,pp) = diag_gmm.anyHitMaxIter;
            GMM_iterMedian(mc,pp) = diag_gmm.iterMedian;
            GMM_iterMax(mc,pp) = diag_gmm.iterMax;
            GMM_finalRelMax(mc,pp) = diag_gmm.finalRelMax;
            GMM_innerConvFrac(mc,pp) = diag_gmm.thetaInnerConvFrac;
            GMM_innerHitCapFrac(mc,pp) = diag_gmm.thetaInnerHitCapFrac;
            GMM_lineSearchFailFrac(mc,pp) = diag_gmm.thetaLineSearchFailFrac;

            %% ------------------------------------------------------------
            % 3. Student-t GPML
            %% ------------------------------------------------------------
            model_student = studentt_gpml_mogp_fit(Xtrain, Ytrain, opts_run);
            Ypred_student = studentt_gpml_mogp_predict(model_student, Xtest);
            RMSE_STUDENT(mc,pp) = sqrt(mean((Ypred_student(:) - YcleanTest(:)).^2));

            %% ------------------------------------------------------------
            % 4. Faithful scalar RCGPR
            %% ------------------------------------------------------------
            model_rcgp = independent_rcgp_gpr_fit(Xtrain, Ytrain, opts_run);
            Ypred_rcgp = independent_rcgp_gpr_predict(model_rcgp, Xtest);
            RMSE_RCGP(mc,pp) = sqrt(mean((Ypred_rcgp(:) - YcleanTest(:)).^2));

            %% ------------------------------------------------------------
            % 5. Standard GPR: current independent-output setup
            %% ------------------------------------------------------------
            model_std = independent_standard_gpr_fit(Xtrain, Ytrain, opts_run);
            Ypred_std = independent_standard_gpr_predict(model_std, Xtest);
            RMSE_STD(mc,pp) = sqrt(mean((Ypred_std(:) - YcleanTest(:)).^2));

            assert(isfield(model_std, 'output') && numel(model_std.output) == p_y, ...
                'Standard GPR must be fit as independent scalar-output models.');

            %% ------------------------------------------------------------
            % 6. Oracle: perfect entry-wise labels + best-guess real-data sigma
            %
            % Important:
            %   - trains on the same contaminated Ytrain;
            %   - knows isOutlierEntry exactly;
            %   - uses oracleInlierObsVar, not the non-oracle sigma_init^2;
            %   - uses oracleOutlierObsVar = oracleInlierObsVar + E[outlier^2].
            %% ------------------------------------------------------------
            opts_oracle = opts_run;
            opts_oracle.oracle_inlier_var  = oracleInlierObsVar;
            opts_oracle.oracle_outlier_var = oracleOutlierObsVar;

            model_oracle = independent_oracle_gmm_style_gpr_fit( ...
                Xtrain, Ytrain, isOutlierEntry, opts_oracle);

            Ypred_oracle = independent_oracle_gmm_style_gpr_predict( ...
                model_oracle, Xtest);

            assert(isfield(model_oracle, 'output') && numel(model_oracle.output) == p_y, ...
                'Oracle must be fit as independent scalar-output models.');

            RMSE_ORACLE(mc,pp) = sqrt(mean((Ypred_oracle(:) - YcleanTest(:)).^2));

            fprintf(['  %s | p_out %.2f | MC %2d/%2d | dims d_x=%d p_y=%d | ', ...
                'outlier entries %4d/%4d | outlier rows %3d/%3d | ', ...
                'Oracle %.4f | ASOR-GPR %.4f | GMM-GPR %.4f | Student-t %.4f | RCGPR %.4f | Standard %.4f\n'], ...
                expCfg.caseName, pOutNow, mc, nMC, d_x, p_y, ...
                nOutEntries, nTotalEntries, nOutRows, nTrain, ...
                RMSE_ORACLE(mc,pp), RMSE_ASOR(mc,pp), RMSE_GMM(mc,pp), ...
                RMSE_STUDENT(mc,pp), RMSE_RCGP(mc,pp), RMSE_STD(mc,pp));

    fprintf('  ASOR: outerConv %d | iterMed %.1f | iterMax %.1f | finalRelMax %.2e | innerConv %.2f | time %.2fs\n', ...
        ASOR_outerConverged(mc,pp), ...
        ASOR_iterMedian(mc,pp), ...
        ASOR_iterMax(mc,pp), ...
        ASOR_finalRelMax(mc,pp), ...
        ASOR_innerConvFrac(mc,pp), ...
        fitTime_ASOR(mc,pp));
    
    fprintf('  GMM : outerConv %d | iterMed %.1f | iterMax %.1f | finalRelMax %.2e | innerConv %.2f | time %.2fs\n', ...
        GMM_outerConverged(mc,pp), ...
        GMM_iterMedian(mc,pp), ...
        GMM_iterMax(mc,pp), ...
        GMM_finalRelMax(mc,pp), ...
        GMM_innerConvFrac(mc,pp), ...
        fitTime_GMM(mc,pp));
        end

        ppElapsed = toc(ppWallTimer);
        fprintf('\n%s | p_out %.2f completed in %.2f minutes using parfor over MC runs.\n', ...
            expCfg.caseName, pOutNow, ppElapsed/60);

    end

    %% ------------------------------------------------------------
    % Store results for this case
    %% ------------------------------------------------------------
    allResults(ee).caseName = expCfg.caseName;
    allResults(ee).caseLabel = expCfg.caseLabel;
    allResults(ee).realDataName = expCfg.realDataName;
    allResults(ee).realDataPath = expCfg.realDataPath;
    allResults(ee).outlierModel = expCfg.outlierModel;
    allResults(ee).outlierScale = expCfg.outlierScale;
    allResults(ee).outlierVar = outlierVar;
    allResults(ee).outlierMean = outlierMean;
    allResults(ee).outlierSecondMoment = outlierSecondMoment;
    allResults(ee).oracleBestGuessSigma = oracleBestGuessSigma;
    allResults(ee).oracleInlierObsVar = oracleInlierObsVar;
    allResults(ee).oracleOutlierObsVar = oracleOutlierObsVar;
    allResults(ee).pOutList = pOutList;
    allResults(ee).sigma_init = sigma_init;
    allResults(ee).nTrain = nTrain;
    allResults(ee).nTest = nTest;
    allResults(ee).nMC = nMC;
    allResults(ee).d_x_store = d_x_store;
    allResults(ee).p_y_store = p_y_store;
    allResults(ee).outlierEntryCount = outlierEntryCount;
    allResults(ee).outlierRowCount = outlierRowCount;
    allResults(ee).methodNames = methodNames;
    allResults(ee).RMSE_ORACLE = RMSE_ORACLE;
    allResults(ee).RMSE_ASOR = RMSE_ASOR;
    allResults(ee).RMSE_GMM = RMSE_GMM;
    allResults(ee).RMSE_STUDENT = RMSE_STUDENT;
    allResults(ee).RMSE_RCGP = RMSE_RCGP;
    allResults(ee).RMSE_STD = RMSE_STD;

    % Updated ASOR/GMM diagnostics.
    allResults(ee).fitTime_ASOR = fitTime_ASOR;
    allResults(ee).fitTime_GMM = fitTime_GMM;
    allResults(ee).ASOR_outerConverged = ASOR_outerConverged;
    allResults(ee).GMM_outerConverged = GMM_outerConverged;
    allResults(ee).ASOR_hitMaxIter = ASOR_hitMaxIter;
    allResults(ee).GMM_hitMaxIter = GMM_hitMaxIter;
    allResults(ee).ASOR_iterMedian = ASOR_iterMedian;
    allResults(ee).GMM_iterMedian = GMM_iterMedian;
    allResults(ee).ASOR_iterMax = ASOR_iterMax;
    allResults(ee).GMM_iterMax = GMM_iterMax;
    allResults(ee).ASOR_finalRelMax = ASOR_finalRelMax;
    allResults(ee).GMM_finalRelMax = GMM_finalRelMax;
    allResults(ee).ASOR_innerConvFrac = ASOR_innerConvFrac;
    allResults(ee).GMM_innerConvFrac = GMM_innerConvFrac;
    allResults(ee).ASOR_innerHitCapFrac = ASOR_innerHitCapFrac;
    allResults(ee).GMM_innerHitCapFrac = GMM_innerHitCapFrac;
    allResults(ee).ASOR_lineSearchFailFrac = ASOR_lineSearchFailFrac;
    allResults(ee).GMM_lineSearchFailFrac = GMM_lineSearchFailFrac;

    dataCell = {RMSE_ORACLE, RMSE_ASOR, RMSE_GMM, RMSE_STUDENT, RMSE_RCGP, RMSE_STD};

    fprintf('\n============================================================\n');
    fprintf('Completed case: %s\n', expCfg.caseName);
    fprintf('Median RMSE summary by p_out:\n');
    for pp = 1:nPout
        fprintf('\np_out = %.2f\n', pOutList(pp));
        for jj = 1:numel(methodNames)
            fprintf('  %-15s median = %.4f | mean = %.4f\n', ...
                methodNames{jj}, ...
                median(dataCell{jj}(:,pp), 'omitnan'), ...
                mean(dataCell{jj}(:,pp), 'omitnan'));
        end
    end
    fprintf('============================================================\n');

fprintf('\nUpdated ASOR/GMM convergence summary for case: %s\n', expCfg.caseName);

for pp = 1:nPout

    fprintf('p_out = %.2f | ASOR conv %.3f hitMax %.3f medIter %.1f medTime %.2fs | GMM conv %.3f hitMax %.3f medIter %.1f medTime %.2fs\n', ...
        pOutList(pp), ...
        mean(ASOR_outerConverged(:,pp)), ...
        mean(ASOR_hitMaxIter(:,pp)), ...
        median(ASOR_iterMax(:,pp), 'omitnan'), ...
        median(fitTime_ASOR(:,pp), 'omitnan'), ...
        mean(GMM_outerConverged(:,pp)), ...
        mean(GMM_hitMaxIter(:,pp)), ...
        median(GMM_iterMax(:,pp), 'omitnan'), ...
        median(fitTime_GMM(:,pp), 'omitnan'));

end

    %% ------------------------------------------------------------
    % Plot immediately after this dataset/outlier case is complete.
    % MATLAB continues to the next case after drawnow.
    %% ------------------------------------------------------------
    figBase = fullfile(outputDir, sprintf('%02d_%s_pout_sweep_boxplot', ee, expCfg.caseName));

    plot_real_case_pout_sweep_boxplot( ...
        dataCell, pOutLabels, methodNames, colors, figBase);

    % Save incremental results after every completed case.
    save(fullfile(outputDir, 'real_data_airquality_Uneg5_15_only_with_oracle_results.mat'), ...
        'allResults', 'experiments', 'methodNames', 'colors', ...
        'nTrain', 'nTest', 'nMC', 'pOutList', 'sigma_init', ...
        'oracleBestGuessSigma', 'oracleBestGuessInlierVar');

    drawnow;
end

%% ========================================================================
% Final summary for the isolated AirQuality case
%% ========================================================================

fprintf('\n\n============================================================\n');
fprintf('FINAL SUMMARY: AirQuality U[-5,15] median RMSE over MC runs\n');
fprintf('============================================================\n');

for ee = activeExperimentIdx
    R = allResults(ee);

    fprintf('\n%s | sigma_init = %.2f | outlierVar = %.4f\n', ...
        R.caseName, R.sigma_init, R.outlierVar);

    vals = {R.RMSE_ORACLE, R.RMSE_ASOR, R.RMSE_GMM, R.RMSE_STUDENT, R.RMSE_RCGP, R.RMSE_STD};

    for pp = 1:nPout
        fprintf('\np_out = %.2f\n', pOutList(pp));
        for jj = 1:numel(methodNames)
            fprintf('  %-15s median = %.4f | mean = %.4f\n', ...
                methodNames{jj}, ...
                median(vals{jj}(:,pp), 'omitnan'), ...
                mean(vals{jj}(:,pp), 'omitnan'));
        end
    end
end

totalElapsed = toc(totalWallTimer);
fprintf('\nTOTAL WALL-CLOCK TIME = %.2f minutes\n', totalElapsed/60);

if useParallelMC
    poolNow = gcp('nocreate');
    if ~isempty(poolNow)
        fprintf('Parallel workers used = %d\n', poolNow.NumWorkers);
    end
end

fprintf('\nSaved figures and MAT file in:\n%s\n', outputDir);
fprintf('============================================================\n');

%% ========================================================================
% Local plotting/stat helpers for this automated real-data script
%% ========================================================================

function [outlierVar, outlierSecondMoment, outlierMean] = outlier_stats_local(outlierModel, outlierScale)
%OUTLIER_STATS_LOCAL
% Returns Var(eta), E[eta^2], and E[eta] for the injected outlier amplitude eta.

    outlierMean = 0;

    switch lower(outlierModel)

        case 'positive_uniform'
            % eta ~ U(0,s)
            outlierMean = outlierScale/2;
            outlierVar = outlierScale^2 / 12;
            outlierSecondMoment = outlierScale^2 / 3;

        case 'negative_uniform'
            % eta ~ U(-s,0)
            outlierMean = -outlierScale/2;
            outlierVar = outlierScale^2 / 12;
            outlierSecondMoment = outlierScale^2 / 3;

        case 'uniform'
            % eta ~ U(-s,s)
            outlierMean = 0;
            outlierVar = outlierScale^2 / 3;
            outlierSecondMoment = outlierScale^2 / 3;

        case 'shifted_uniform'
            % eta ~ U(a,b), with outlierScale = [a,b]
            a = outlierScale(1);
            b = outlierScale(2);
            outlierMean = 0.5*(a + b);
            outlierVar = (b - a)^2 / 12;
            outlierSecondMoment = (a^2 + a*b + b^2) / 3;

        case 'gaussian'
            % eta ~ N(0, outlierScale^2)
            outlierMean = 0;
            outlierVar = outlierScale^2;
            outlierSecondMoment = outlierScale^2;

        case 'positive_shift'
            outlierMean = outlierScale;
            outlierVar = 0;
            outlierSecondMoment = outlierScale^2;

        case 'negative_shift'
            outlierMean = -outlierScale;
            outlierVar = 0;
            outlierSecondMoment = outlierScale^2;

        otherwise
            error('Unknown outlierModel: %s', outlierModel);
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

function model = asor_mogp_fit(X, Y, opts)
    [n, d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    % Simple non-robust initialization from contaminated data.
    % No MAD, no robustStd, no robust preprocessing.
    m_hat = median(Y,1)';

    % All non-oracle methods must receive the same fixed incorrect R.
    if ~isfield(opts, 'init_R')
        error('ASOR requires opts.init_R. Do not initialize R from data.');
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
            y_i = Y(i,:)';
            diff_i = y_i - mu_i;
            R(i) = diff_i' * SigmaInv * diff_i + trace(SigmaInv * V_i);
            beta(i) = b_hat + 0.5*R(i);
            logRatio = log((1-opts.theta0)/opts.theta0) + gammaln(alpha_shape) ...
                     - gammaln(opts.a) + opts.a*log(max(b_hat, realmin)) ...
                     - alpha_shape*log(max(beta(i), realmin)) + 0.5*R(i);
            Omega(i) = logistic_inverse_from_log_ratio(logRatio);
            w(i) = Omega(i) + (1 - Omega(i)) * alpha_shape / beta(i);
            w(i) = max(w(i), opts.minWeight);
            S_blocks{i} = diff_i*diff_i' + V_i;
        end
        S = zeros(p,p);
        for i = 1:n
            S = S + w(i)*S_blocks{i};
        end

        % ------------------------------------------------------------
        % Covariance / R update
        % ------------------------------------------------------------
        if ~isfield(opts, 'asor_R_update')
            opts.asor_R_update = 'iw_mode_original';
        end
        
        switch lower(opts.asor_R_update)
        
            case 'iw_mode_original'
                % Original ASOR VB/IW-mode update:
                %
                % Sigma = (sum_i w_i S_i + S0)/(n + nu0 + p + 1)
                denom_R = n + opts.nu0 + p + 1;
        
            case 'weighted_denominator'
                % Experimental ASOR_GPR_EXP update:
                %
                % Sigma = (sum_i w_i S_i + S0)/(sum_i w_i + nu0 + p + 1)
                %
                % This tests whether ASOR's p_out=0 underestimation of sigma
                % is caused by weighted numerator / unweighted denominator.
                denom_R = sum(w) + opts.nu0 + p + 1;
        
            otherwise
                error('Unknown opts.asor_R_update = %s', opts.asor_R_update);
        end
        
        Sigma_hat = (S + S0) / max(denom_R, realmin);
        Sigma_hat = make_spd(Sigma_hat, opts.jitter);
        
        numerator = opts.A - 1 + opts.a * sum(1 - Omega);
        denominator = opts.B + sum((1 - Omega) .* (alpha_shape ./ beta));
        b_hat = max(numerator / max(denominator, realmin), realmin);
        
        Lm = kron(ones(n,1), eye(p));
        m_hat = (Lm' * Kinv * Lm) \ (Lm' * Kinv * mu_f);
        
        if opts.learnTheta && it > opts.thetaBurnIn && mod(it, opts.thetaUpdateEvery) == 0
            objFun = @(lt) kernel_variational_objective(lt, X, mu_f, Sigma_f, m_hat, p, opts.jitter);
            fopts = optimset('Display','off', 'MaxFunEvals', opts.thetaMaxFunEvals, 'MaxIter', opts.thetaMaxIter);
            logtheta = fminsearch(objFun, logtheta, fopts);
            Kx = ard_rbf_kernel(X, X, logtheta, opts.jitter);
            Ktheta = kron(Kx, eye(p));
        end
        
        weightedResidualObj = sum(w .* R);
        relChange = abs(prevObj - weightedResidualObj) / max(1, abs(prevObj));
        if it > opts.thetaBurnIn && relChange < opts.tol
            break;
        end
        prevObj = weightedResidualObj;
    end
    model = struct('X',X, 'Y',Y, 'n',n, 'p',p, ...
    'mu_f',mu_f, ...
    'Sigma_f',Sigma_f, ...
    'm_hat',m_hat, ...
    'logtheta',logtheta, ...
    'Kinv',inv(make_spd(Ktheta, opts.jitter)), ...
    'w',w, ...
    'Omega',Omega, ...
    'Sigma_hat',Sigma_hat, ...
    'b_hat',b_hat);
end

function model = gmm_mogp_fit(X, Y, opts)
    [n, d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    % Mean initialized only from contaminated data
    m_hat = median(Y,1)';

    % ------------------------------------------------------------
    % Fair fixed non-oracle covariance initialization
    % ------------------------------------------------------------
    if ~isfield(opts, 'init_R')
        error('GMM requires opts.init_R. Do not initialize R from data.');
    end
    
    Sigma_nom0 = make_spd(opts.init_R, opts.jitter);
    
    % Outlier covariance initialized as inflated version of the same fixed R.
    % This is not estimated from data.
    % FAIRNESS FIX: No massive inflation hint. 
    % Both components start near the same wrong naive guess.
    % We use a tiny 10% perturbation purely to break symmetry for EM.
    Sigma_out0 = make_spd(1.1 * Sigma_nom0, opts.jitter);
    Sigma_nom0 = make_spd(0.9 * Sigma_nom0, opts.jitter);
    
    % Convention:
    %   component 1 = outlier / broad component
    %   component 2 = nominal / narrow component
    Sigma_comp = {Sigma_out0; Sigma_nom0};
    alpha = [0.5, 0.5];

    logtheta = [log(opts.init_lengthscale*ones(d,1)); log(opts.init_signal_std)];

    mu_f = kron(ones(n,1), m_hat);
    Sigma_f = eye(n*p);
    gamma = repmat(alpha, n, 1);

    prevObj = inf;

    for it = 1:opts.maxIter

        % ------------------------------------------------------------
        % Build expected likelihood precision
        % ------------------------------------------------------------
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

        % ------------------------------------------------------------
        % Latent GP posterior update
        % ------------------------------------------------------------
        Kx = ard_rbf_kernel(X, X, logtheta, opts.jitter);
        Ktheta = make_spd(kron(Kx, eye(p)), opts.jitter);
        Kinv = inv(Ktheta);

        Precision_f = make_spd(Kinv + Lambda, opts.jitter);
        Sigma_f = inv(Precision_f);

        mf = kron(ones(n,1), m_hat);
        mu_f = Sigma_f * (Kinv*mf + Lambda*yvec);

        % ------------------------------------------------------------
        % Responsibility update
        % IMPORTANT FIX: include trace(invSigma * posterior covariance)
        % ------------------------------------------------------------
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

        % ------------------------------------------------------------
        % Mixing weight update
        % ------------------------------------------------------------
        alpha = max(mean(gamma,1), 1e-8);
        alpha = alpha / sum(alpha);

        % ------------------------------------------------------------
        % Component covariance update
        % ------------------------------------------------------------
        for j = 1:2
            S_j = zeros(p,p);
            sum_gamma = sum(gamma(:,j));
        
            for i = 1:n
                idx = sample_block(i,p);
                diff_i = Y(i,:)' - mu_f(idx);
                V_i = Sigma_f(idx,idx);
        
                S_j = S_j + gamma(i,j) * (diff_i*diff_i' + V_i);
            end
        
            % ------------------------------------------------------------
            % Optional weak covariance prior for GMM
            %
            % Component convention:
            %   j = 1 : outlier / broad component
            %   j = 2 : nominal / narrow component
            %
            % This makes GMM use the same prior-scale philosophy as ASOR:
            %   nominal prior variance around sigma_asor_prior^2
            %   outlier prior variance around sigma_asor_prior^2 * init_b/a
            %
            % Importantly, this does NOT change the shared wrong init_R.
            % It only regularizes the EM covariance update.
            % ------------------------------------------------------------
            if isfield(opts, 'gmm_use_cov_prior') && opts.gmm_use_cov_prior
        
                if j == 1
                    prior_strength = opts.gmm_prior_strength_out;
                    prior_var = opts.gmm_outlier_prior_var;
                else
                    prior_strength = opts.gmm_prior_strength_nom;
                    prior_var = opts.gmm_nominal_prior_var;
                end
        
                prior_strength = max(prior_strength, 0);
                prior_cov = prior_var * eye(p);
        
                Sigma_comp{j} = make_spd( ...
                    (S_j + prior_strength * prior_cov) / ...
                    max(sum_gamma + prior_strength, realmin), ...
                    opts.jitter);
        
            else
                Sigma_comp{j} = make_spd(S_j / max(sum_gamma, realmin), opts.jitter);
            end
        end

        % ------------------------------------------------------------
        % Enforce label convention:
        % component 1 = broader/outlier, component 2 = narrower/nominal
        % ------------------------------------------------------------
        if trace(Sigma_comp{2}) > trace(Sigma_comp{1})
            tmpS = Sigma_comp{1};
            Sigma_comp{1} = Sigma_comp{2};
            Sigma_comp{2} = tmpS;

            alpha = fliplr(alpha);
            gamma = fliplr(gamma);
        end

        % ------------------------------------------------------------
        % Mean update
        % ------------------------------------------------------------
        Lm = kron(ones(n,1), eye(p));
        m_hat = (Lm' * Kinv * Lm) \ (Lm' * Kinv * mu_f);

        % ------------------------------------------------------------
        % Kernel hyperparameter update
        % Same gradient-descent optimizer as ASOR-GPR
        % ------------------------------------------------------------
        if opts.learnTheta && it > opts.thetaBurnIn && mod(it, opts.thetaUpdateEvery) == 0
        
            logtheta_old = logtheta;
        
            [logtheta, thetaInfo] = kernel_variational_gradient_descent( ...
                logtheta, X, mu_f, Sigma_f, m_hat, p, opts);
        
            if opts.thetaGradVerbose
                fprintf('GMM-GPR-Grad iter %d | J %.6g -> %.6g | grad %.3e | step %.3e\n', ...
                    it, thetaInfo.J0, thetaInfo.Jfinal, thetaInfo.gradNorm, thetaInfo.lastStep);
            end
        
            % Safety fallback: if gradient update fails, keep old theta.
            if any(~isfinite(logtheta))
                logtheta = logtheta_old;
            end
        end

        % ------------------------------------------------------------
        % Convergence diagnostic
        % ------------------------------------------------------------
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

        relChange = abs(prevObj - obj) / max(1, abs(prevObj));

        if it > opts.thetaBurnIn && relChange < opts.tol
            break;
        end

        prevObj = obj;
    end

    Kx_final = ard_rbf_kernel(X, X, logtheta, opts.jitter);
    Ktheta_final = make_spd(kron(Kx_final, eye(p)), opts.jitter);

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
        'gamma',gamma);
end



function model = rcgp_mogp_fit(X, Y, opts)
% Multi-Output RCGP. Learns a shared covariance R and a single weight w_i per sample.
    [n, d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    % Simple non-robust initialization from contaminated data.
    % No MAD, no robustStd, no robust preprocessing.
    m_hat = median(Y,1)';

    % All non-oracle methods must receive the same fixed incorrect R.
    if ~isfield(opts, 'init_R')
        error('RCGP requires opts.init_R. Do not initialize R from data.');
    end

    R = make_spd(opts.init_R, opts.jitter);
    
    epsilon = opts.rcgp_epsilon;
    w = ones(n,1);
    logtheta = [log(opts.init_lengthscale*ones(d,1)); log(opts.init_signal_std)];
    
    mu_f = kron(ones(n,1), m_hat);
    Sigma_f = eye(n*p);
    prevObj = inf;
    for it = 1:opts.maxIter
        Rinv = inv(R);
        Lambda = kron(diag(w), Rinv);
        
        Kx = ard_rbf_kernel(X, X, logtheta, opts.jitter);
        Ktheta = make_spd(kron(Kx, eye(p)), opts.jitter);
        Kinv = inv(Ktheta);
        
        Precision_f = make_spd(Kinv + Lambda, opts.jitter);
        Sigma_f = inv(Precision_f);
        mf = kron(ones(n,1), m_hat);
        mu_f = Sigma_f * (Kinv*mf + Lambda*yvec);
        
        Rquad = zeros(n,1);
        S_all = zeros(p,p,n);
        for i = 1:n
            idx = sample_block(i,p);
            yi = Y(i,:)'; mui = mu_f(idx); Vi = Sigma_f(idx,idx);
            ri = yi - mui;
            Rquad(i) = ri' * Rinv * ri + trace(Rinv * Vi);
            S_all(:,:,i) = ri*ri' + Vi;
        end
        
        c2 = max(quantile(Rquad, 1 - epsilon), 1e-8);
        w_new = min(max(1 ./ sqrt(1 + Rquad ./ c2), 1e-4), 1.0);
        w = 0.5*w + 0.5*w_new; % Damping
        
        S = zeros(p,p);
        for i = 1:n
            S = S + w(i) * S_all(:,:,i);
        end
        R = nearest_spd(0.5*R + 0.5*(S / n)); % Update Covariance
        
        Lm = kron(ones(n,1), eye(p));
        m_hat = (Lm' * Kinv * Lm) \ (Lm' * Kinv * mu_f);
        
        %% ------------------------------------------------------------
        % Kernel hyperparameter update
        % Same gradient-descent optimizer as ASOR/GMM
        %% ------------------------------------------------------------
        if opts.learnTheta && it > opts.thetaBurnIn && mod(it, opts.thetaUpdateEvery) == 0
        
            logtheta_old = logtheta;
        
            [logtheta, thetaInfo] = kernel_variational_gradient_descent( ...
                logtheta, X, mu_f, Sigma_f, m_hat, p, opts);
        
            if opts.thetaGradVerbose
                fprintf('RCGPR iter %d | J %.6g -> %.6g | grad %.3e | step %.3e\n', ...
                    it, thetaInfo.J0, thetaInfo.Jfinal, thetaInfo.gradNorm, thetaInfo.lastStep);
            end
        
            if any(~isfinite(logtheta))
                logtheta = logtheta_old;
            end
        end
        
        obj = sum(Rquad);
        relChange = abs(prevObj - obj) / max(1, abs(prevObj));
        if it > opts.thetaBurnIn && relChange < opts.tol
            break;
        end
        prevObj = obj;
    end
    model = struct('X',X, 'Y',Y, 'n',n, 'p',p, ...
    'mu_f',mu_f, ...
    'Sigma_f',Sigma_f, ...
    'm_hat',m_hat, ...
    'logtheta',logtheta, ...
    'Kinv',inv(make_spd(kron(ard_rbf_kernel(X, X, logtheta, opts.jitter), eye(p)), opts.jitter)), ...
    'R_hat',R, ...
    'w',w);
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

function val = standard_mogp_nlml(z, X, Y, d, p, jitter)

    n = size(X,1);
    yvec = stack_samples(Y);

    logtheta = z(1:d+1);

    % Hard bounds to prevent pathological kernels.
    ell = exp(logtheta(1:d));
    sf  = exp(logtheta(end));

    Xrange = max(X,[],1) - min(X,[],1);
    Xrange(Xrange <= 0 | ~isfinite(Xrange)) = 1;

    if any(ell < 0.02*Xrange(:)) || any(ell > 5.0*Xrange(:)) || ...
       sf < 1e-4 || sf > 1e4
        val = 1e20;
        return;
    end

    if p == 1
        sn = exp(z(d+2));
        if sn < 1e-5 || sn > 1e3
            val = 1e20;
            return;
        end
        R = sn^2;
    else
        diagVals = exp(z(d+2:end));
        if any(diagVals < 1e-5) || any(diagVals > 1e3)
            val = 1e20;
            return;
        end
        Lr = diag(diagVals);
        R = Lr*Lr';
    end

    R = make_spd(R, jitter);

    Kx = ard_rbf_kernel(X, X, logtheta, jitter);
    Ktheta = kron(Kx, eye(p));
    Ky = make_spd(Ktheta + kron(eye(n), R), jitter);

    [L,flag] = chol(Ky,'lower');
    if flag ~= 0
        val = 1e20;
        return;
    end

    Lm = kron(ones(n,1), eye(p));

    KyInvY = L' \ (L \ yvec);
    KyInvM = L' \ (L \ Lm);

    A_m = Lm' * KyInvM;
    b_m = Lm' * KyInvY;

    m_hat = A_m \ b_m;
    mf = kron(ones(n,1), m_hat);

    r = yvec - mf;
    alpha = L' \ (L \ r);

    val = 0.5*(r'*alpha) + sum(log(diag(L))) + 0.5*numel(yvec)*log(2*pi);

    if ~isfinite(val)
        val = 1e20;
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


function [XN, mu, sd] = standardize_train_local_student_gpml(X)
% Local standardization helper to avoid naming conflicts.

    mu = mean(X,1);
    sd = std(X,0,1);
    sd(sd < 1e-12 | ~isfinite(sd)) = 1;
    XN = (X - mu) ./ sd;
end


function hypcov = init_cov_ard_student_gpml(X, y)
%INIT_COV_ARD_STUDENT_GPML
% Initial hyperparameters for GPML covSEard.

    d = size(X,2);

    % Since X is standardized, lengthscale 1 is a reasonable initial value.
    log_ell = zeros(d,1);

    % Since y is standardized, signal std 1 is reasonable.
    log_sf = log(std(y) + 1e-6);

    hypcov = [log_ell; log_sf];
end

function model = standard_mogp_fit(X, Y, opts)
%STANDARD_MOGP_FIT Plain Gaussian multi-output GP baseline.
%
% This is the standard GP baseline:
%   y_i = f_i + e_i,   e_i ~ N(0, R0)
%
% Important:
%   - R0 is fixed.
%   - R0 is NOT SigmaTrue.
%   - R0 comes from opts.init_R, which is a fixed intentionally incorrect value.
%   - Only kernel parameters and mean are fitted.

    [n,d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    %% ------------------------------------------------------------
    % Mean initialization
    %% ------------------------------------------------------------
    m0 = mean(Y,1)';

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
    [logtheta, stdInfo] = standard_fixedR_nlml_gradient_descent( ...
        logtheta0, X, Y, R_fixed, opts);
    
    bestVal = standard_mogp_fixedR_nlml(logtheta, X, Y, R_fixed, opts.jitter);

    % %% ------------------------------------------------------------
    % % Optimize only kernel parameters: lengthscales and signal std
    % %% ------------------------------------------------------------
    % starts = zeros(5,numel(logtheta0));
    % starts(1,:) = logtheta0(:)';
    % starts(2,:) = logtheta0(:)' + [log(0.7)*ones(1,d), log(1.0)];
    % starts(3,:) = logtheta0(:)' + [log(1.3)*ones(1,d), log(1.0)];
    % starts(4,:) = logtheta0(:)' + [log(1.0)*ones(1,d), log(0.7)];
    % starts(5,:) = logtheta0(:)' + [log(1.0)*ones(1,d), log(1.3)];
    % 
    % fopts = optimset('Display','off', ...
    %                  'MaxFunEvals', max(250, opts.thetaMaxFunEvals), ...
    %                  'MaxIter', max(250, opts.thetaMaxIter), ...
    %                  'TolX', 1e-5, ...
    %                  'TolFun', 1e-5);
    % 
    % bestVal = inf;
    % bestTheta = logtheta0;
    % 
    % for s = 1:size(starts,1)
    %     lt0 = starts(s,:)';
    % 
    %     try
    %         [lt_try, val_try] = fminsearch( ...
    %             @(lt) standard_mogp_fixedR_nlml(lt, X, Y, R_fixed, opts.jitter), ...
    %             lt0, fopts);
    % 
    %         if isfinite(val_try) && val_try < bestVal
    %             bestVal = val_try;
    %             bestTheta = lt_try;
    %         end
    %     catch
    %         % Ignore bad restart
    %     end
    % end
    % 
    % logtheta = bestTheta;

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
    model.alpha_exact = alphaVec;  % <--- ADD THIS LINE HERE
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
function model = oracle_mogp_fit_trueR(X, Y, isOutlier, opts)
    keep = ~isOutlier(:);
    if sum(keep) < 5
        keep = true(size(isOutlier(:)));
    end
    X = X(keep,:);
    Y = Y(keep,:);
    [n, d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);
    R = make_spd(opts.oracle_R, opts.jitter);
    Rinv = inv(R);
    m_hat = median(Y,1)';
    logtheta = [log(opts.init_lengthscale*ones(d,1)); log(opts.init_signal_std)];
    Sigma_f = eye(n*p);
    mu_f = yvec;
    for it = 1:opts.maxIter
        Kx = ard_rbf_kernel(X, X, logtheta, opts.jitter);
        Ktheta = make_spd(kron(Kx, eye(p)), opts.jitter);
        Kinv = inv(Ktheta);
        Lambda = kron(eye(n), Rinv);
        Precision_f = make_spd(Kinv + Lambda, opts.jitter);
        Sigma_f = inv(Precision_f);
        mf = kron(ones(n,1), m_hat);
        mu_f = Sigma_f * (Kinv*mf + Lambda*yvec);
        Lm = kron(ones(n,1), eye(p));
        m_hat = (Lm' * Kinv * Lm) \ (Lm' * Kinv * mu_f);
        if opts.learnTheta && it > opts.thetaBurnIn && mod(it, opts.thetaUpdateEvery) == 0
            objFun = @(lt) kernel_variational_objective(lt, X, mu_f, Sigma_f, m_hat, p, opts.jitter);
            fopts = optimset('Display','off', 'MaxFunEvals', opts.thetaMaxFunEvals, 'MaxIter', opts.thetaMaxIter);
            logtheta = fminsearch(objFun, logtheta, fopts);
        end
    end
    model = struct('X',X, 'Y',Y, 'n',n, 'p',p, 'mu_f',mu_f, 'Sigma_f',Sigma_f, ...
        'm_hat',m_hat, 'logtheta',logtheta, 'Kinv',inv(make_spd(kron(ard_rbf_kernel(X, X, logtheta, opts.jitter), eye(p)), opts.jitter)));
end

%% ========================================================================
% SHARED PREDICTION AND KERNELS
% ========================================================================
function [Ypred, VarPred] = shared_mogp_predict(model, Xtest)
    Xtrain = model.X;
    n = model.n;
    p = model.p;
    nTest = size(Xtest,1);
    Kinv = model.Kinv;
    mu_f = model.mu_f;
    Sigma_f = model.Sigma_f;
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
        
        % <--- REPLACE THE mu_star LINE WITH THIS BLOCK --->
        if isfield(model, 'alpha_exact')
            mu_star = m + KstarX * model.alpha_exact;
        else
            mu_star = m + KstarX*Kinv*(mu_f - mf);
        end
        % <------------------------------------------------>
        
        Sigma_star = Kss - KstarX*Kinv*KstarX' + KstarX*Kinv*Sigma_f*Kinv*KstarX';
        Sigma_star = 0.5*(Sigma_star + Sigma_star');
        Ypred(t,:) = mu_star';
        VarPred(:,:,t) = Sigma_star;
    end
end

function J = kernel_variational_objective(logtheta, X, mu_f, Sigma_f, m_hat, p, jitter)
    n = size(X,1);
    Kx = ard_rbf_kernel(X, X, logtheta, jitter);
    Ktheta = make_spd(kron(Kx, eye(p)), jitter);
    mf = kron(ones(n,1), m_hat);
    diff = mu_f - mf;
    [R, flag] = chol(Ktheta);
    if flag ~= 0, J = 1e20; return; end
    J = real(0.5 * diff' * (R \ (R' \ diff)) + 0.5 * trace(R \ (R' \ Sigma_f)) + sum(log(diag(R))));
    if ~isfinite(J), J = 1e20; end
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
function draw_grouped_filled_boxplots_boxchart(dataCell, xLevels, methodNames, colors)
    nMethods = numel(dataCell);
    nP = numel(xLevels);
    nMC = size(dataCell{1},1);
    centers = 1:nP;
    delta = 0.10;
    offs = ((1:nMethods) - mean(1:nMethods)) * delta;
    hold on;
    bw = 0.09;
    h = gobjects(nMethods,1);
    allVals = [];
    for j = 1:nMethods
        Xpos = repelem(centers + offs(j), nMC).';
        Yvec = reshape(dataCell{j}, [], 1);         % <--- REMOVED THE TRANSPOSE
        allVals = [allVals; Yvec]; %#ok<AGROW>
        h(j) = boxchart(Xpos, Yvec, 'BoxWidth', bw, 'BoxFaceColor', colors(j,:), 'BoxFaceAlpha', 0.60);
        h(j).LineWidth = 1.2;
    end
    for s = 1:(nP-1)
        xline(s + 0.5, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.4, 'HandleVisibility', 'off');
    end
    yMax = 1.1 * max([allVals; 0.1], [], 'omitnan');
    xlim([0.5, nP+0.5]); ylim([0, yMax]);
    xticks(centers); xticklabels(string(xLevels));
    legend(h, methodNames, 'Location', 'northwest', 'AutoUpdate', 'off', 'FontSize', 20);
end

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

function Ainv = inv_spd(A)
    A = make_spd(A, 1e-10);
    [~,L] = chol(A);
    Ainv = (L' \ (L \ eye(size(A))));
    Ainv = (Ainv + Ainv')/2;
end

function A = nearest_spd(A)
%NEAREST_SPD Simple symmetric positive-definite projection.
    % Symmetrize the matrix
    A = (A + A')/2;
    
    % Perform eigendecomposition
    [V,D] = eig(A);
    d = diag(D);
    
    % Project any negative or zero eigenvalues to a small positive number
    d = max(real(d), 1e-10);
    
    % Reconstruct the matrix
    A = V*diag(d)*V';
    
    % Ensure exact symmetry after reconstruction
    A = real((A + A')/2);
end

function R0 = robust_initial_R_from_data(Y, jitter)

    [~,p] = size(Y);

    medY = median(Y,1);
    madY = median(abs(Y - medY),1);
    robustStd = 1.4826 * madY(:);

    empiricalStd = std(Y,0,1)';
    bad = ~isfinite(robustStd) | robustStd < 1e-6;
    robustStd(bad) = max(empiricalStd(bad), 0.1);

    % Conservative non-oracle noise guess from observed Y.
    sigma0 = max(0.20 * robustStd, 0.05);

    R0 = diag(sigma0.^2);

    if p > 1
        Yc = Y - median(Y,1);
        C = cov(Yc);

        if all(isfinite(C(:))) && rank(C) == p
            C = make_spd(C, jitter);
            D = sqrt(diag(C));
            Corr = C ./ max(D*D', realmin);
            Corr = max(min(Corr, 0.5), -0.5);
            Corr(1:p+1:end) = 1;
            R0 = diag(sigma0) * Corr * diag(sigma0);
        end
    end

    R0 = make_spd(R0, jitter);
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
            % Symmetric uniform outliers U[-outlierScale, outlierScale].
            outlierNoise = outlierScale * (2*rand(n,p) - 1);

        case 'shifted_uniform'
            % General shifted uniform outliers U[a,b].
            % Here outlierScale must be [a,b].
            a = outlierScale(1);
            b = outlierScale(2);
            outlierNoise = a + (b - a) * rand(n,p);

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

function model = independent_asor_gpr_fit(X, Y, opts)
%INDEPENDENT_ASOR_GPR_FIT
% Fits one scalar-output ASOR-GPR model per output dimension.
%
% This converts ASOR from vector-level robustness:
%     one weight for y_i in R^p
%
% to independent scalar-output robustness:
%     one separate ASOR model for each y_{i,q}
%
% This is better matched to entry-wise independent outliers.

    [n, d] = size(X);
    p = size(Y,2);

    model = struct();
    model.X = X;
    model.n = n;
    model.d = d;
    model.p = p;
    model.output = cell(p,1);
    model.method = 'Independent-ASOR-GPR';

    if ~isfield(opts, 'init_R')
        error('independent_asor_gpr_fit requires opts.init_R.');
    end

    for q = 1:p

        opts_q = opts;

        % Each scalar ASOR model receives the corresponding scalar
        % diagonal entry of the same fixed incorrect covariance.
        opts_q.init_R = opts.init_R(q,q);

        % For scalar-output ASOR, p = 1 inside asor_mogp_fit_grad.
        % Therefore the inverse-Wishart degree parameter should also
        % be scalar-output-consistent.
        if isfield(opts_q, 'nu0_base_offset')
            opts_q.nu0 = 1 + opts_q.nu0_base_offset;
        else
            opts_q.nu0 = 3;
        end

        % Fit scalar ASOR-GPR to output q.
        model.output{q} = asor_mogp_fit_grad(X, Y(:,q), opts_q);
    end
end

function Ypred = independent_asor_gpr_predict(model, Xtest)
%INDEPENDENT_ASOR_GPR_PREDICT
% Predicts all output dimensions from independent scalar ASOR-GPR models.

    nTest = size(Xtest,1);
    p = model.p;

    Ypred = zeros(nTest, p);

    for q = 1:p
        [yq_pred, ~] = shared_mogp_predict(model.output{q}, Xtest);
        Ypred(:,q) = yq_pred(:,1);
    end
end




function model = asor_mogp_fit_grad(X, Y, opts)
%ASOR_MOGP_FIT_GRAD
% Same ASOR-GPR model as asor_mogp_fit, but kernel hyperparameters
% are updated using analytic-gradient descent on the ASOR variational
% kernel objective J_theta.

    [n, d] = size(X);
    p = size(Y,2);
    yvec = stack_samples(Y);

    % Contaminated-data mean initialization
    m_hat = median(Y,1)';

    % Same fixed incorrect covariance initialization as other non-oracle methods
    if ~isfield(opts, 'init_R')
        error('ASOR-GPR-Grad requires opts.init_R. Do not initialize R from data.');
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
        
        %% ------------------------------------------------------------
        % Covariance / sigma^2 update
        %% ------------------------------------------------------------
        if ~isfield(opts, 'asor_R_update')
            opts.asor_R_update = 'iw_mode_original';
        end
        
        switch lower(opts.asor_R_update)
        
            case 'iw_mode_original'
                % Original ASOR VB/IW-mode update:
                %
                % Sigma = (sum_i w_i S_i + S0)/(n + nu0 + p + 1)
                denom_R = n + opts.nu0 + p + 1;
                Sigma_hat = (S + S0) / max(denom_R, realmin);
        
            case 'weighted_denominator'
                % Experimental IW-style weighted denominator:
                %
                % Sigma = (sum_i w_i S_i + S0)/(sum_i w_i + nu0 + p + 1)
                denom_R = sum(w) + opts.nu0 + p + 1;
                Sigma_hat = (S + S0) / max(denom_R, realmin);
        
            case 'scalar_ig_conjugate_mean'
                % Proper scalar inverse-gamma conjugate update:
                %
                % sigma^2 ~ IG(alpha0, beta0)
                %
                % alpha_N = alpha0 + n/2
                % beta_N  = beta0 + 0.5 * sum_i E[I_i] E[(y_i-f_i)^2]
                %
                % Since p = 1 here:
                % S = sum_i w_i * ((y_i-mu_i)^2 + V_i)
                %
                % posterior mean:
                % sigma^2_hat = beta_N / (alpha_N - 1)
        
                if p ~= 1
                    error('scalar_ig_conjugate_mean is only valid for scalar-output ASOR, p = 1.');
                end
        
                if ~isfield(opts, 'alpha0_sigma') || ~isfield(opts, 'beta0_sigma')
                    error('scalar_ig_conjugate_mean requires opts.alpha0_sigma and opts.beta0_sigma.');
                end
        
                alpha_N_sigma = opts.alpha0_sigma + 0.5*n;
                beta_N_sigma  = opts.beta0_sigma  + 0.5*S(1,1);
        
                sigma2_hat = beta_N_sigma / max(alpha_N_sigma - 1, realmin);
        
                Sigma_hat = sigma2_hat;
        
            case 'scalar_ig_conjugate_map'
                % Same inverse-gamma posterior, but use MAP instead of posterior mean:
                %
                % sigma^2_MAP = beta_N / (alpha_N + 1)
        
                if p ~= 1
                    error('scalar_ig_conjugate_map is only valid for scalar-output ASOR, p = 1.');
                end
        
                if ~isfield(opts, 'alpha0_sigma') || ~isfield(opts, 'beta0_sigma')
                    error('scalar_ig_conjugate_map requires opts.alpha0_sigma and opts.beta0_sigma.');
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

        %% ------------------------------------------------------------
        % Analytic-gradient kernel update
        %% ------------------------------------------------------------
        if opts.learnTheta && it > opts.thetaBurnIn && mod(it, opts.thetaUpdateEvery) == 0

            logtheta_old = logtheta;

            [logtheta, thetaInfo] = kernel_variational_gradient_descent( ...
                logtheta, X, mu_f, Sigma_f, m_hat, p, opts);

            if opts.thetaGradVerbose
                fprintf('ASOR-GPR-Grad iter %d | J %.6g -> %.6g | grad %.3e | step %.3e\n', ...
                    it, thetaInfo.J0, thetaInfo.Jfinal, thetaInfo.gradNorm, thetaInfo.lastStep);
            end

            % Safety fallback: if gradient update fails, keep old theta.
            if any(~isfinite(logtheta))
                logtheta = logtheta_old;
            end

            Kx = ard_rbf_kernel(X, X, logtheta, opts.jitter);
            Ktheta = kron(Kx, eye(p));
        end

        weightedResidualObj = sum(w .* R);
        relChange = abs(prevObj - weightedResidualObj) / max(1, abs(prevObj));

        if it > opts.thetaBurnIn && relChange < opts.tol
            break;
        end

        prevObj = weightedResidualObj;
    end

    Ktheta_final = make_spd(kron(ard_rbf_kernel(X, X, logtheta, opts.jitter), eye(p)), opts.jitter);

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
        'method','ASOR-GPR-Grad', ...
        'asor_R_update', opts.asor_R_update);
end


function model = independent_gmm_gpr_fit(X, Y, opts)
%INDEPENDENT_GMM_GPR_FIT
% Fits one scalar-output GMM-GPR model per output dimension.
%
% This converts GMM-GPR from vector-level mixture responsibilities:
%     gamma_i for y_i in R^p
%
% to independent scalar-output mixture responsibilities:
%     gamma_{i,q} through a separate scalar GMM-GPR per output q.
%
% This is better matched to entry-wise independent outliers.

    [n, d] = size(X);
    p = size(Y,2);

    model = struct();
    model.X = X;
    model.n = n;
    model.d = d;
    model.p = p;
    model.output = cell(p,1);
    model.method = 'Independent-GMM-GPR';

    if ~isfield(opts, 'init_R')
        error('independent_gmm_gpr_fit requires opts.init_R.');
    end

    for q = 1:p

        opts_q = opts;

        % Each scalar GMM model receives the corresponding scalar
        % diagonal entry of the same fixed incorrect covariance.
        opts_q.init_R = opts.init_R(q,q);

        % Fit scalar GMM-GPR to output q.
        model.output{q} = gmm_mogp_fit(X, Y(:,q), opts_q);
    end
end

function Ypred = independent_gmm_gpr_predict(model, Xtest)
%INDEPENDENT_GMM_GPR_PREDICT
% Predicts all output dimensions from independent scalar GMM-GPR models.

    nTest = size(Xtest,1);
    p = model.p;

    Ypred = zeros(nTest, p);

    for q = 1:p
        [yq_pred, ~] = shared_mogp_predict(model.output{q}, Xtest);
        Ypred(:,q) = yq_pred(:,1);
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
    
    gmmOutlierInitMultiplier = 50;
    gmmNominalInitMultiplier = 1;
    
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

            if isfield(opts, 'gmm_use_cov_prior') && opts.gmm_use_cov_prior

                if j == 1
                    prior_strength = opts.gmm_prior_strength_out;
                    prior_var = opts.gmm_outlier_prior_var;
                else
                    prior_strength = opts.gmm_prior_strength_nom;
                    prior_var = opts.gmm_nominal_prior_var;
                end

                prior_strength = max(prior_strength, 0);
                prior_cov = prior_var * eye(p);

                Sigma_comp{j} = make_spd( ...
                    (S_j + prior_strength * prior_cov) / ...
                    max(sum_gamma + prior_strength, realmin), ...
                    opts.jitter);

            else
                Sigma_comp{j} = make_spd(S_j / max(sum_gamma, realmin), opts.jitter);
            end
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
