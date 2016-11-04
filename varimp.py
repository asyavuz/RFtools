#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Variable importance measures for ensemble objects.

This module currently only includes a permutation importance measure, called PIMP.

- The `PIMP` class handles the permutation importance measure calculated via permutation of response variables.
"""

# Author: Ahmet Sinan Yavuz <asinanyavuz@sabanciuniv.edu>
# BSD-3 License
import math
import numpy as np
import warnings
from scipy import stats
from sklearn.utils import check_X_y


class PIMP(object):
    """A permutation variable importance measure for ensemble classifiers.

    PIMP methodology repeatedly permutes response variable in a dataset to calculate an unbiased variable importance
    measure in the form of a p-value [1].

    Parameters
    ----------
    estimator : object
        The ensemble estimator object with 'fit' function and feature_importances_ attribute. Original implementation
        of PIMP uses Gini importance as feature importance measure.

    s : integer, optional (default=100)
        Number of permutations to perform in order to get null distribution of variable importance measure.

    seed : integer, optional (default=323)
        Seed for random number generator of numpy.

    use_normalised_imp : bool, optional (default=False)
         An boolean variable that determines whether to use normalised feature importance values or not.
         If you want to be consistent with the original implementation, please keep this parameter False.

    verbose : bool, optional (default=False)
         Controls verbosity of output. If set ``False`, nothing will be printed to screen.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        Feature importance values of the original fit. Higher values indicate higher importance.

    permuted_feature_importances_ : array of shape = [s, n_features]
        Feature importance values of the permuted runs.

    permutation_count : int
        Number of permutations performed to estimate null distribution. If permute function was never run with
        warm_start=True, this value should be equal to s parameter after completion of a run.

    p_values_ : array of shape = [n_features]
        Calculated p-values of features, indicating probability of a feature importance occurring randomly.
        It can be calculated with a given null distribution or calculated empirically. In order to obtain p-values,
        you need to run calculate_pvals function with appropriate parameters first.

    support_ : array of shape = [n_features]
        The mask of features that have p-values lower than specified p-value threshold.

    p_value_dist_ : str or list
        Background distribution that is used for p-value calculation. If feature-wise distributions are calculated
        or provided, this attribute will give the list of distributions for each feature.

    References
    ----------
    .. [1] A. Altmann et al. "Permutation importance: a corrected feature importance measure",
           Bioinformatics, 26(10), 2010, pp. 1340–1347.
    """

    def __init__(self, estimator=None, s=100, seed=323, use_normalised_imp=False, verbose=False):
        # Basic sanity checks
        if hasattr(estimator, 'warm_start'):
            if estimator.warm_start:
                raise ValueError("Warm start parameter of the estimator should be set False for repeated fresh "
                                 "training with the estimator object.")

        assert s >= 0

        # Core attributes
        self.estimator = estimator
        self.s = s
        self.seed = seed
        self.use_normalised_imp = use_normalised_imp
        self.verbose = verbose

        # If there is one, use already calculated feature importances for speed-up.
        if hasattr(estimator, 'feature_importances_'):
            self.feature_importances_ = self.estimator.feature_importances_
        else:
            self.feature_importances_ = None

        # Attributes that will be filled in after appropriate functions run
        self.permuted_feature_importances_ = None
        self.permutation_count = 0
        self.p_values_ = None
        self.support_ = None
        self.p_value_dist_ = None

        # Original parameters set when calculate_pvals was called. Preserved for potential combination with other
        # PIMP objects.
        self.p_value_method_ = None

    def permute(self, X, y, warm_start=False):
        """This function permutes the response variable for ``s`` times and stores the results
        in ``object.permuted_feature_importances_`` attribute.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            The training input. It will be converted into numpy array internally by scikit-learn.

        y: array-like, shape = [n_samples]
            The response variable. This is the class value for classification tasks.

        warm_start: bool, optional (default=False)
            When set ``True`` permutation results will be appended into already calculated results from previous run of
            ``permute()`` function, otherwise, it discards any old results if there is any.

        Returns
        -------
        self : object
            Returns self.
        """
        # Basic parameter checks
        if self.verbose:
            print("Performing basic sanity checks on provided data...")
        X, y = check_X_y(X, y)

        if self.verbose:
            print("-Done.")

        # Fit original data to obtain untouched feature importance values
        if self.feature_importances_ is None:
            if self.verbose:
                print("Training model without any permutation...")
            try:
                clf = self.estimator.fit(X, y)
            except Exception as e:
                raise ValueError(
                    "Error in fitting the classifier. Please check your estimator has a fit function or the provided "
                    "data cannot be fitted to the provided classifier.\n" + str(e))

            if self.verbose:
                print("-Done.")

            try:
                if self.use_normalised_imp:
                    self.feature_importances_ = clf.feature_importances_
                else:
                    self.feature_importances_ = np.zeros(clf.feature_importances_.shape, dtype=np.float)
                    for tree in clf.estimators_:
                        self.feature_importances_ += tree.tree_.compute_feature_importances(normalize=False)

                    self.feature_importances_ /= len(clf.estimators_)
            except Exception as e:
                raise ValueError("Only methods with feature_importance_ attribute is supported in PIMP.\n" + str(e))

        perm_feat_imps = None

        # Start permuting label array
        if self.verbose:
            print("Starting permutation of y...")

        np.random.seed(self.seed)
        for curr_itr in range(1, self.s + 1):
            # Permute label vector
            y_perm = np.random.permutation(y)

            # Train with permuted label vector
            clf = self.estimator.fit(X, y_perm)

            # Record feature importances after permutation
            if self.use_normalised_imp:
                curr_feat_imp = clf.feature_importances_
            else:
                curr_feat_imp = np.zeros(clf.feature_importances_.shape, dtype=np.float)
                for tree in clf.estimators_:
                    curr_feat_imp += tree.tree_.compute_feature_importances(normalize=False)

                curr_feat_imp /= len(clf.estimators_)

            if curr_itr > 1:
                perm_feat_imps = np.vstack([perm_feat_imps, curr_feat_imp])
            else:
                perm_feat_imps = curr_feat_imp

            if self.verbose:
                print("-Permutation %d of %d completed." % (curr_itr, self.s))

        if warm_start:
            self.permuted_feature_importances_ = np.vstack([self.permuted_feature_importances_, perm_feat_imps])
        else:
            self.permuted_feature_importances_ = perm_feat_imps

        self.permutation_count += self.s
        if self.verbose:
            print("Permutation step completed.")

        return self

    def calculate_pvals(self, method="nonpar", two_sided=True, logp=False, min_size=10, thr=0.05, null_dists=None,
                        **kwargs):
        """This function calculates p-values for each feature based on non-parametric estimation, user provided null
        distribution or a null distribution determined using Kolmogorov-Smirnov test. Maximum likelihood estimators of
        the parameters of the selected distribution are computed.

        A single null distribution can be used for all features or feature-wise null distribution can be set
        or determined.

        Calculated p-values are stored in the ``object.p_values_`` attribute. Null distribution(s) used for calculation
        of p-values can be viewed via ``object.p_value_dist_`` attribute.

        Parameters
        ----------
        method: str, optional (default="nonpar")
            Method for calculating p-values. Accepted values are "nonpar" for non-parametric estimation, "norm" for
            normal distribution, "lognorm" for lognormal distribution, "gamma" for gamma distribution, "ks" for fitting
            a null distribution via Kolmogorov-Smirnov tests, "ks_each" for fitting a separate null distribution for
            each feature via Kolmogorov-Smirnov tests, and "user_each" for providing a null distibution for each
            feature manually. For the last case, you should provide an array-like object with ``null_dists`` parameter.
            Given object should have a shape of ``[n_features]`` and for each feature one of the four options
            ("norm", "lnorm", "gamma", "nonpar") should be provided.

        two_sided: bool, optional (default=True)
            When set ``True``, p-values will be calculated using both left and right tail of the null distribution.

        logp: bool, optional (default=False)
            When set ``True``, p-values will be stored as -log10(p-value).

        min_size: int, optional (default=10)
            Minimum required number of positive observations to compute maximum likelihood estimators of the parameters
            of the selected distribution. This parameter will not be used if method is set to ``norm`` or ``nonpar``.

        thr: float, optional (default=0.05)
            Threshold p-value for significance.

        null_dists: array-like, shape=[n_features] or None
            Null distributions to be used for each feature. Only used if method is set to ``user_each``.

        Returns
        -------
        self : object
            Returns self.
        """
        # Basic parameter checks
        if self.feature_importances_ is None:
            raise ValueError("Feature importances are not available. Please run permute function first!")

        if self.permuted_feature_importances_ is None:
            raise ValueError("Permuted feature importances are not available. Please run permute function first!")

        if method not in ["ks", "ks_each", "nonpar", "normal", "lognormal", "gamma", "user_each"]:
            raise ValueError("Unsupported method: %s. Please choose among options: 'ks', 'nonpar', 'normal', "
                             "'lognormal', 'gamma'. See documentation for details." % method)

        if method == "user_each" and null_dists is None:
            raise ValueError("In order to use 'user_each' method, you need to provide null distributions to be used "
                             "for each feature using null_dists parameter.")

        if method == "ks_each" and null_dists is not None:
            warnings.warn("Null distributions provided in null_dists parameter will be ignored as the method selected "
                          "as ks_each.", RuntimeWarning)

        # Start p-value calculation
        if self.verbose:
            print("Starting p-value calculation (selected method: %s)..." % method)

        # Log run settings for later use
        self.p_value_method_ = {"method": method, "two_sided": two_sided, "logp": logp, "min_size": min_size,
                                "thr": thr, "kwargs": kwargs}

        if method != "user_each":
            self.p_value_dist_ = method
        else:
            self.p_value_dist_ = null_dists

        n_feats = self.feature_importances_.shape[0]

        # Compute maximum likelihood estimators of distribution parameters
        if method in ["norm", "ks"]:
            means_norm = np.mean(self.permuted_feature_importances_, axis=0)
            stds_norm = np.std(self.permuted_feature_importances_, axis=0, ddof=1)  # ddof=1 is set 1 to get sample
            # standard deviation as original code does in R)

        if method in ["lognorm", "ks"]:
            means_lnorm = np.zeros(self.permuted_feature_importances_.shape[1], dtype=np.float)
            stds_lnorm = np.zeros(self.permuted_feature_importances_.shape[1], dtype=np.float)

            pos_obss = np.sum(self.permuted_feature_importances_ > 0, axis=0)

            for curr_feat in range(0, n_feats):
                if pos_obss[curr_feat] >= min_size:
                    pos_rows = self.permuted_feature_importances_[:, curr_feat] > 0

                    # Mean of positive observations
                    means_lnorm[curr_feat] = np.mean(np.log(self.permuted_feature_importances_[pos_rows, curr_feat]))

                    # Sd of positive observations (ddof=1 is set 1 to get sample standard deviation as original
                    # code does in R)
                    stds_lnorm[curr_feat] = np.std(np.log(self.permuted_feature_importances_[pos_rows, curr_feat]),
                                                   ddof=1)
                else:
                    means_lnorm[curr_feat] = np.NaN
                    stds_lnorm[curr_feat] = np.NaN

        if method in ["gamma", "ks"]:
            s_gamma = np.zeros(self.permuted_feature_importances_.shape[1], dtype=np.float)
            means_gamma = np.zeros(self.permuted_feature_importances_.shape[1], dtype=np.float)
            pos_obss = np.sum(self.permuted_feature_importances_ > 0, axis=0)
            for curr_feat in range(0, n_feats):
                if pos_obss[curr_feat] >= min_size:
                    pos_rows = self.permuted_feature_importances_[:, curr_feat] > 0
                    s_gamma[curr_feat] = np.log(
                        np.mean(self.permuted_feature_importances_[pos_rows, curr_feat])) - np.mean(
                        np.log(self.permuted_feature_importances_[pos_rows, curr_feat]))

                    # This case is separated to be same with original code.
                    # In original code, for means "> min_size" is required not ">= min_size")
                    if pos_obss[curr_feat] > min_size:
                        means_gamma[curr_feat] = np.mean(self.permuted_feature_importances_[pos_rows, curr_feat])
                    else:
                        means_gamma[curr_feat] = np.NaN
                else:
                    s_gamma[curr_feat] = np.NaN

            k_gamma = (3 - s_gamma + np.sqrt((s_gamma - 3) ** 2 + 24 * s_gamma)) / (12 * s_gamma)
            scale_gamma = 1 / k_gamma * means_gamma
            shape_gamma = k_gamma

        # Kolmogorov-Smirnov Goodness of Fit test to determine appropriate distribution
        if method == "ks" or method == "ks_each":
            p_norms = np.zeros(n_feats, dtype=np.float)
            p_lognorms = np.zeros(n_feats, dtype=np.float)
            p_gammas = np.zeros(n_feats, dtype=np.float)

            for curr_feat in range(0, n_feats):
                pos_rows = self.permuted_feature_importances_[:, curr_feat] > 0
                if means_norm[curr_feat] == 0 and stds_norm[curr_feat] == 0:
                    # There is no check for this case in scipy.cdf functions. Gives out a runtime warning.
                    p_norms[curr_feat] = np.NaN
                else:
                    d_norm, p_norm = stats.kstest(self.permuted_feature_importances_[:, curr_feat], 'norm',
                                                  (means_norm[curr_feat], stds_norm[curr_feat]))
                    p_norms[curr_feat] = p_norm

                if np.isnan(means_lnorm[curr_feat] + stds_lnorm[curr_feat]):
                    p_lognorms[curr_feat] = np.NaN
                else:
                    d_lognorm, p_lognorm = stats.kstest(self.permuted_feature_importances_[pos_rows, curr_feat],
                                                        'lognorm', (stds_lnorm[curr_feat], 0,
                                                                    math.exp(means_lnorm[curr_feat])))
                    p_lognorms[curr_feat] = p_lognorm

                if np.isnan(scale_gamma[curr_feat] + shape_gamma[curr_feat]):
                    p_gammas[curr_feat] = np.NaN
                else:
                    d_gamma, p_gamma = stats.kstest(self.permuted_feature_importances_[pos_rows, curr_feat],
                                                    'gamma', (shape_gamma[curr_feat], 0, scale_gamma[curr_feat]))
                    p_gammas[curr_feat] = p_gamma

                if method == "ks_each":
                    ks_pvals = [p_norms[curr_feat], p_lognorms[curr_feat], p_gammas[curr_feat]]

                    max_p = max(ks_pvals)
                    ind_sel = ks_pvals.index(max_p)
                    selected_method = None
                    if ind_sel == 0:
                        selected_method = "norm"
                    elif ind_sel == 1:
                        selected_method = "lognorm"
                    elif ind_sel == 2:
                        selected_method = "gamma"

                    # Use user-provided p-value cutoff for Kolmogorov-Smirnov test if available
                    if "ks_pval" in kwargs.keys():
                        ks_cutoff = float(kwargs["ks_pval"])
                    else:
                        ks_cutoff = 0.01  # Original code uses 0.01 threshold

                    if max_p <= ks_cutoff:
                        null_dists[curr_feat] = "nonpar"
                    else:
                        null_dists[curr_feat] = selected_method
            if method == "ks_each":
                self.p_value_dist_ = null_dists  # Update the attribute with new selected distributions
                method = "user_each"
            else:
                # The 0.05 quantile (=5% percentile) p-value for each distribution is computed.
                q_norm = np.nanpercentile(p_norms, 5)
                q_lognorm = np.nanpercentile(p_lognorms, 5)
                q_gamma = np.nanpercentile(p_gammas, 5)

                ks_pvals = [q_norm, q_lognorm, q_gamma]
                max_p = max(ks_pvals)
                ind_sel = ks_pvals.index(max_p)
                selected_method = None
                if ind_sel == 0:
                    selected_method = "norm"
                elif ind_sel == 1:
                    selected_method = "lognorm"
                elif ind_sel == 2:
                    selected_method = "gamma"

                # Use user-provided p-value threshold for Kolmogorov-Smirnov test if available
                if "ks_pval" in kwargs.keys():
                    ks_cutoff = float(kwargs["ks_pval"])
                else:
                    ks_cutoff = 0.01  # Original code uses 0.01 threshold

                if max_p <= ks_cutoff:
                    method = "nonpar"
                else:
                    method = selected_method

                self.p_value_dist_ = method  # Update the attribute with the new selected distribution
                if self.verbose:
                    print("Kolmogorov-Smirnov Goodness of Fit Test Results:")
                    print("------------------------------------------------")
                    print("| Distribution         | 5% percentile P-value |")
                    print("------------------------------------------------")
                    print("| Normal Distribution  | %21.5g |" % q_norm)
                    print("| Lognorm Distribution | %21.5g |" % q_lognorm)
                    print("| Gamma Distribution   | %21.5g |" % q_gamma)
                    print("------------------------------------------------")
                    print("Selected method: %s" % method)

        pvals = np.zeros(n_feats, dtype=np.float)

        if method == "normal":
            for curr_feat in range(0, n_feats):
                pvals[curr_feat], curr_dist = self._norm_pval(self.feature_importances_[curr_feat],
                                                   means_norm[curr_feat],
                                                   stds_norm[curr_feat], two_sided=two_sided, logp=logp)

        elif method == "lognormal":
            for curr_feat in range(0, n_feats):
                pvals[curr_feat], curr_dist = self._lognorm_pval(self.feature_importances_[curr_feat],
                                                      self.permuted_feature_importances_[:, curr_feat],
                                                      stds_lnorm[curr_feat],
                                                      means_lnorm[curr_feat], two_sided=two_sided, logp=logp)

        elif method == "gamma":
            for curr_feat in range(0, n_feats):
                pvals[curr_feat], curr_dist = self._gamma_pval(self.feature_importances_[curr_feat],
                                                    self.permuted_feature_importances_[:, curr_feat],
                                                    shape_gamma[curr_feat],
                                                    scale_gamma[curr_feat], two_sided=two_sided, logp=logp)

        if method == "nonpar":
            for curr_feat in range(0, n_feats):
                pvals[curr_feat], curr_dist = self._nonpar_pval(self.feature_importances_[curr_feat],
                                                     self.permuted_feature_importances_[:, curr_feat],
                                                     two_sided=two_sided,
                                                     logp=logp)

        if method == "user_each":
            for curr_feat in range(0, n_feats):
                if method == "normal":
                    for curr_feat in range(0, n_feats):
                        pvals[curr_feat], curr_dist = self._norm_pval(self.feature_importances_[curr_feat],
                                                                      means_norm[curr_feat],
                                                                      stds_norm[curr_feat], two_sided=two_sided,
                                                                      logp=logp)

                elif method == "lognormal":
                    for curr_feat in range(0, n_feats):
                        pvals[curr_feat], curr_dist = self._lognorm_pval(self.feature_importances_[curr_feat],
                                                                         self.permuted_feature_importances_[:,
                                                                         curr_feat],
                                                                         stds_lnorm[curr_feat],
                                                                         means_lnorm[curr_feat], two_sided=two_sided,
                                                                         logp=logp)

                elif method == "gamma":
                    for curr_feat in range(0, n_feats):
                        pvals[curr_feat], curr_dist = self._gamma_pval(self.feature_importances_[curr_feat],
                                                                       self.permuted_feature_importances_[:, curr_feat],
                                                                       shape_gamma[curr_feat],
                                                                       scale_gamma[curr_feat], two_sided=two_sided,
                                                                       logp=logp)
                # Return to non-parametric if unknown distribution is inputted.
                else:
                    for curr_feat in range(0, n_feats):
                        pvals[curr_feat], curr_dist = self._nonpar_pval(self.feature_importances_[curr_feat],
                                                                        self.permuted_feature_importances_[:,
                                                                        curr_feat],
                                                                        two_sided=two_sided,
                                                                        logp=logp)

                # In order to store the actual method used in p-value calculation for each feature, p_value_dist_ is
                # updated here to reflect possible runtime method changes, due to failed computation of maximum
                # likelihood estimators of the selected distribution parameters. This is not done only for this case,
                # as it'll needlessly consume memory to store repeatedly the same information for the single
                # distribution case.
                self.p_value_dist_[curr_feat] = curr_dist

        self.p_values_ = pvals
        self.support_ = pvals <= thr

        return self

    def combine(self, other, safe=False):
        """This function combines two or more PIMP objects, merges null distributions and re-calculates
        p-values, if it is calculated in the original object.

        Parameters
        ----------
        other: object or list of objects
            One or a list of multiple PIMP objects to be combined with this PIMP object.

        safe: bool, optional (default=False)
            When set ``True``, if feature_importances_ attributes of the new objects is not equal to original object,
            merging objects will raise an error. Setting this parameter ``True`` requires other PIMP objects fitted
            with same seed in estimator object.

        Returns
        -------
        self : object
            Returns self.
        """

        if isinstance(other, list):
            for i, curr_other in enumerate(other):
                if not isinstance(curr_other, PIMP):
                    raise ValueError("Combining requires PIMP objects. "
                                     "You have provided a %s object (index: %d)." % (type(curr_other), i))
                if safe:
                    if np.array_equal(self.feature_importances_, curr_other.feature_importances_):
                        self.permuted_feature_importances_ = np.vstack([self.permuted_feature_importances_,
                                                                        curr_other.permuted_feature_importances_])
                        self.permutation_count += curr_other.permutation_count
                    else:
                        raise ValueError("Other PIMP (index: %d) object does not have an identical feature_importances_"
                                         " attribute! With safe parameter set True, you cannot merge two different "
                                         "runs." % i)
                else:
                    # At least we should check whether feature counts are equal.
                    if self.permuted_feature_importances_.shape[1] == curr_other.permuted_feature_importances_.shape[1]:
                        self.permuted_feature_importances_ = np.vstack([self.permuted_feature_importances_,
                                                                        curr_other.permuted_feature_importances_])
                        self.permutation_count += curr_other.permutation_count
                    else:
                        raise ValueError("Combining two PIMP objects requires runs with same feature set!")

            if self.p_values_ is not None:
                self.calculate_pvals(self.p_value_method_["method"],
                                     self.p_value_method_["two_sided"],
                                     self.p_value_method_["logp"],
                                     self.p_value_method_["min_size"],
                                     self.p_value_method_["thr"],
                                     **self.p_value_method_["kwargs"])

        elif isinstance(other, PIMP):
            if safe:
                if np.array_equal(self.feature_importances_, other.feature_importances_):
                    self.permuted_feature_importances_ = np.vstack([self.permuted_feature_importances_,
                                                                    other.permuted_feature_importances_])
                    self.permutation_count += other.permutation_count
                else:
                    raise ValueError("Other PIMP object does not have an identical feature_importances_"
                                     " attribute! With safe parameter set True, you cannot merge two different "
                                     "runs.")
            else:
                # At least we should check whether feature counts are equal.
                if self.permuted_feature_importances_.shape[1] == other.permuted_feature_importances_.shape[1]:
                    self.permuted_feature_importances_ = np.vstack([self.permuted_feature_importances_,
                                                                    other.permuted_feature_importances_])
                    self.permutation_count += other.permutation_count
                else:
                    raise ValueError("Combining two PIMP objects requires runs with same feature set!")
            if self.p_values_ is not None:
                self.calculate_pvals(self.p_value_method_["method"],
                                     self.p_value_method_["two_sided"],
                                     self.p_value_method_["logp"],
                                     self.p_value_method_["min_size"],
                                     self.p_value_method_["thr"],
                                     **self.p_value_method_["kwargs"])
        else:
            raise ValueError("Combining requires PIMP objects. You have provided a %s object." % type(other))

        return self

    def _nonpar_pval(self, imp, rnd, two_sided=True, logp=False):
        # One-sided p-value (upper tail)
        pv = sum(rnd >= imp) / float(len(rnd))

        # If two-sided is requested, get the lower tail as well
        if two_sided:
            pv = min(pv, sum(rnd <= imp) / float(len(rnd)))

        # Transform to -log10(p-value)
        if logp:
            pv = -math.log10(pv)

        return pv, "nonpar"

    def _norm_pval(self, imp, mean, sd, two_sided=True, logp=False):
        # One-sided p-value (upper tail)
        pv = 1 - stats.norm.cdf(imp, loc=mean, scale=sd)

        # If two-sided is requested, get the other tail as well
        if two_sided:
            pv = min(pv, 1 - pv)

        # Transform to -log10(p-value)
        if logp:
            pv = -math.log10(pv)

        return pv, "norm"

    def _lognorm_pval(self, imp, rnd, mean, sd, two_sided=True, logp=False):
        if np.isnan(mean + sd):  # parameters could not be estimated
            return self._nonpar_pval(imp, rnd, two_sided=two_sided, logp=logp), "nonpar"
        else:  # lognormal fitting
            # One-sided p-value (upper tail)
            pv = 1 - stats.lognorm.cdf(imp, sd, scale=math.exp(mean))

            # If two-sided is requested, get the other tail as well
            if two_sided:
                pv = min(pv, 1 - pv)

            # Transform to -log10(p-value)
            if logp:
                pv = -math.log10(pv)

            return pv, "lognorm"

    def _gamma_pval(self, imp, rnd, shape, scale, two_sided=True, logp=False):
        if np.isnan(shape + scale):  # parameters could not be estimated
            return self._nonpar_pval(imp, rnd, two_sided=two_sided, logp=logp), "nonpar"
        else: # gamma fitting
            # One-sided p-value (upper tail)
            pv = 1 - stats.gamma.cdf(imp, shape, scale=scale)

            # If two-sided is requested, get the other tail as well
            if two_sided:
                pv = min(pv, 1 - pv)

            # Transform to -log10(p-value)
            if logp:
                pv = -math.log10(pv)

            return pv, "gamma"
