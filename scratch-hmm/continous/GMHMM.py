from scratchmm import _BaseHMM

class GMMHMM(_BaseHMM):
    """Hidden Markov Model with Gaussin mixture emissions

    Attributes
    ----------
    n_components : int (read-only)
        Number of states in the model.
    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.
    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.
    gmms: array of GMM objects, length 'n_components`
        GMM emission distributions for each state

    Methods
    -------
    eval(X)
        Compute the log likelihood of `X` under the HMM.
    decode(X)
        Find most likely state sequence for each point in `X` using the
        Viterbi algorithm.
    rvs(n=1)
        Generate `n` samples from the HMM.
    init(X)
        Initialize HMM parameters from `X`.
    fit(X)
        Estimate HMM parameters from `X` using the Baum-Welch algorithm.
    predict(X)
        Like decode, find most likely state sequence corresponding to `X`.
    score(X)
        Compute the log likelihood of `X` under the model.

    Examples
    --------
    >>> from sklearn.hmm import GMMHMM
    >>> GMMHMM(n_components=2, n_mix=10, cvtype='diag')
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    GMMHMM(cvtype='diag',
        gmms=[GMM(cvtype='diag', n_components=10), GMM(cvtype='diag', n_components=10)],
        n_components=2, n_mix=10, startprob=array([ 0.5,  0.5]),
        startprob_prior=1.0,
        transmat=array([[ 0.5,  0.5],
           [ 0.5,  0.5]]),
        transmat_prior=1.0)


    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_components=1, n_mix=1, startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 gmms=None, cvtype=None, var=3):
        """Create a hidden Markov model with GMM emissions.

        Parameters
        ----------
        n_components : int
            Number of states.
        """
        super(GMMHMM, self).__init__(n_components, startprob, transmat,
                                     startprob_prior=startprob_prior,
                                     transmat_prior=transmat_prior)

        # XXX: Hotfit for n_mix that is incompatible with the scikit's
        # BaseEstimator API
        self.n_mix = n_mix
        self.cvtype = cvtype
	self.var = var
        if gmms is None:
            gmms = []
            for x in xrange(self.n_components):
                if cvtype is None:
                    g = GMM(n_mix)
                else:
                    g = GMM(n_mix, cvtype=cvtype)
                gmms.append(g)
        self.gmms = gmms

    def _compute_log_likelihood(self, obs):
        return np.array([g.score(obs) for g in self.gmms]).T

    def _generate_sample_from_state(self, state, random_state=None):
        return self.gmms[state].rvs(1, random_state=random_state).flatten()

    def _init(self, obs, params='stwmc'):
        super(GMMHMM, self)._init(obs, params=params)

        allobs = np.concatenate(obs, 0)
	n_centers = self.n_components*self.n_mix
	cluster_centers = cluster.KMeans(n_clusters=n_centers).fit(allobs).cluster_centers_
	K = cluster_centers.shape[1]
	inds = random.sample(np.arange(0,n_centers),n_centers)

	for i in xrange(self.n_components):
            self.gmms[i].n_features = K

	if 'm' in params:
	    for i in xrange(self.n_components):
                self.gmms[i]._means = cluster_centers[inds[(i*self.n_mix):((i+1)*self.n_mix)],:] + np.random.multivariate_normal(np.zeros(K),np.eye(K)*self.var,self.n_mix)

        if 'c' in params:
            cv = np.cov(obs[0].T)
            if not cv.shape:
                cv.shape = (1, 1)
            for i in xrange(self.n_components):
                self.gmms[i]._covars = _distribute_covar_matrix_to_match_cvtype(
                    cv, self.cvtype, self.n_mix)

    def _initialize_sufficient_statistics(self):
        stats = super(GMMHMM, self)._initialize_sufficient_statistics()
        stats['norm'] = [np.zeros(g.weights.shape) for g in self.gmms]
        stats['means'] = [np.zeros(np.shape(g.means)) for g in self.gmms]
        stats['covars'] = [np.zeros(np.shape(g._covars)) for g in self.gmms]
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(GMMHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        for state, g in enumerate(self.gmms):
            lgmm_posteriors = np.log(g.eval(obs)[1] + np.finfo(np.float).eps)
            lgmm_posteriors += np.log(posteriors[:, state][:, np.newaxis]
                                      + np.finfo(np.float).eps)
            gmm_posteriors = np.exp(lgmm_posteriors)
            tmp_gmm = GMM(g.n_components, cvtype=g.cvtype)
            tmp_gmm.n_features = g.n_features
            tmp_gmm.covars = _distribute_covar_matrix_to_match_cvtype(
                                np.eye(g.n_features), g.cvtype, g.n_components)
            norm = tmp_gmm._do_mstep(obs, gmm_posteriors, params)

            if np.any(np.isnan(tmp_gmm.covars)):
                raise ValueError

            stats['norm'][state] += norm
            if 'm' in params:
                stats['means'][state] += tmp_gmm.means * norm[:, np.newaxis]
            if 'c' in params:
                if tmp_gmm.cvtype == 'tied':
                    stats['covars'][state] += tmp_gmm._covars * norm.sum()
                else:
                    cvnorm = np.copy(norm)
                    shape = np.ones(tmp_gmm._covars.ndim)
                    shape[0] = np.shape(tmp_gmm._covars)[0]
                    cvnorm.shape = shape
                    stats['covars'][state] += tmp_gmm._covars * cvnorm

    def _do_mstep(self, stats, params, covars_prior=1e-2, **kwargs):
        super(GMMHMM, self)._do_mstep(stats, params)
        # All that is left to do is to apply covars_prior to the
        # parameters updated in _accumulate_sufficient_statistics.
        for state, g in enumerate(self.gmms):
            norm = stats['norm'][state]
            if 'w' in params:
                g.weights = normalize(norm)
            if 'm' in params:
                g.means = stats['means'][state] / norm[:, np.newaxis]
            if 'c' in params:
                if g.cvtype == 'tied':
                    g.covars = ((stats['covars'][state]
                                 + covars_prior * np.eye(g.n_features))
                                / norm.sum())
                else:
                    cvnorm = np.copy(norm)
                    shape = np.ones(g._covars.ndim)
                    shape[0] = np.shape(g._covars)[0]
                    cvnorm.shape = shape
                    if g.cvtype == 'spherical' or g.cvtype == 'diag':
                        g.covars = (stats['covars'][state]
                                    + covars_prior) / cvnorm
                    elif g.cvtype == 'full':
                        eye = np.eye(g.n_features)
                        g.covars = ((stats['covars'][state]
                                     + covars_prior * eye[np.newaxis, :, :])
                                    / cvnorm)
