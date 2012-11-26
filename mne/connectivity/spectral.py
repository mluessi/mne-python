# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from warnings import warn
from inspect import getargspec

import numpy as np
from scipy.fftpack import fftfreq

import logging
logger = logging.getLogger('mne')


from .utils import check_indices
from ..parallel import parallel_func
from .. import Epochs, SourceEstimate
from ..time_frequency.multitaper import dpss_windows, _mt_spectra,\
                                        _psd_from_mt, _csd_from_mt,\
                                        _psd_from_mt_adaptive
from .. import verbose

########################################################################
# Various connectivity estimators


class EpochMeanConEstBase(object):
    """Base class for methods that estimate connectivity as mean over epochs"""
    def __init__(self, n_cons, n_freqs):
        self.n_cons = n_cons
        self.n_freqs = n_freqs
        self.con_scores = None

    def start_epoch(self):
        """This method is called at the start of each epoch"""
        pass  # for this type of con. method we don't do anything

    def combine(self, other):
        """Include con. accumated for some epochs in this estimate"""
        self._acc += other._acc


class CohEstBase(EpochMeanConEstBase):
    """Base Estimator for Coherence, Coherency, Imag. Coherence"""
    def __init__(self, n_cons, n_freqs):
        super(CohEstBase, self).__init__(n_cons, n_freqs)

        # allocate space for accumulation of CSD
        self._acc = np.zeros((n_cons, n_freqs), dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate CSD for some connections"""
        self._acc[con_idx] += csd_xy


class CohEst(CohEstBase):
    """Coherence Estimator"""
    name = 'Coherence'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):
        """Compute final con. score for some connections"""
        if self.con_scores is None:
            self.con_scores = np.zeros((self.n_cons, self.n_freqs))
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.abs(csd_mean)\
                                   / np.sqrt(psd_xx * psd_yy)


class CohyEst(CohEstBase):
    """Coherency Estimator"""
    name = 'Coherency'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):
        """Compute final con. score for some connections"""
        if self.con_scores is None:
            self.con_scores = np.zeros((self.n_cons, self.n_freqs))
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = csd_mean\
                                   / np.sqrt(psd_xx * psd_yy)


class ImCohEst(CohEstBase):
    """Imaginary Coherence Estimator"""
    name = 'Imaginary Coherence'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):
        """Compute final con. score for some connections"""
        if self.con_scores is None:
            self.con_scores = np.zeros((self.n_cons, self.n_freqs))
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.imag(csd_mean)\
                                   / np.sqrt(psd_xx * psd_yy)


class PLIEst(EpochMeanConEstBase):
    """PLI Estimator"""
    name = 'PLI'

    def __init__(self, n_cons, n_freqs):
        super(PLIEst, self).__init__(n_cons, n_freqs)
        self._acc = np.zeros((n_cons, n_freqs))

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections"""
        self._acc[con_idx] += np.sign(np.imag(csd_xy))

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections"""
        if self.con_scores is None:
            self.con_scores = np.zeros((self.n_cons, self.n_freqs))
        pli_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = pli_mean


class WPLIEst(EpochMeanConEstBase):
    """WPLI Estimator"""
    name = 'WPLI'

    def __init__(self, n_cons, n_freqs):
        super(WPLIEst, self).__init__(n_cons, n_freqs)
        #store  both imag(csd) and abs(imag(csd))
        self._acc = np.zeros((n_cons, 2 * n_freqs))

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections"""
        im_csd = np.imag(csd_xy)
        self._acc[con_idx] += np.c_[im_csd, np.abs(im_csd)]

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections"""
        if self.con_scores is None:
            self.con_scores = np.zeros((self.n_cons, self.n_freqs))
        acc_mean = self._acc[con_idx] / n_epochs
        num = np.abs(acc_mean[:, :self.n_freqs])
        denom = acc_mean[:, self.n_freqs:]
        self.con_scores[con_idx] = num / denom


class WPLIDebiasedEst(EpochMeanConEstBase):
    """Debiased WPLI Square Estimator"""
    name = 'Debiased WPLI Square'

    def __init__(self, n_cons, n_freqs):
        super(WPLIDebiasedEst, self).__init__(n_cons, n_freqs)
        #store imag(csd), abs(imag(csd)), imag(csd)^2
        self._acc = np.zeros((n_cons, 3 * n_freqs))

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections"""
        im_csd = np.imag(csd_xy)
        self._acc[con_idx] += np.c_[im_csd, np.abs(im_csd),
                                    im_csd ** 2]

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections"""
        if self.con_scores is None:
            self.con_scores = np.zeros((self.n_cons, self.n_freqs))

        # note: we use the trick from fieldtrip to compute the
        # the estimate over all pairwise epoch combinations
        n = self.n_freqs
        sum_im_csd = self._acc[con_idx, :n]
        sum_abs_im_csd = self._acc[con_idx, n: 2 * n]
        sum_sq_im_csd = self._acc[con_idx, 2 * n:]

        con = (sum_im_csd ** 2 - sum_sq_im_csd)\
              / (sum_abs_im_csd ** 2 - sum_sq_im_csd)

        self.con_scores[con_idx] = con


########################################################################


def _epoch_spectral_connectivity(data, sfreq, dpss, eigvals, freq_mask,
                                 adaptive, idx_map, block_size, psd,
                                 accumulate_psd, con_method_types, con_methods,
                                 accumulate_inplace=True):
    n_freqs = np.sum(freq_mask)
    n_cons = len(idx_map[0])

    """Connectivity estimation for one epoch see spectral_connectivity"""
    if not accumulate_inplace:
        # instantiate methods only for this epoch (used in parallel mode)
        con_methods = [mtype(n_cons, n_freqs) for mtype in con_method_types]

    # compute tapered spectra
    x_mt, _ = _mt_spectra(data, dpss, sfreq)

    if adaptive:
        # compute PSD and adaptive weights
        this_psd, weights = _psd_from_mt_adaptive(x_mt, eigvals, freq_mask,
                                                  return_weights=True)

        # only keep freqs of interest
        x_mt = x_mt[:, :, freq_mask]
    else:
        # do not use adaptive weights
        x_mt = x_mt[:, :, freq_mask]
        weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
        this_psd = _psd_from_mt(x_mt, weights)

    # accumulate or return psd
    if accumulate_psd:
        if accumulate_inplace:
            psd += this_psd
        else:
            psd = this_psd
    else:
        psd = None

    # tell the methods that a new epoch starts
    for method in con_methods:
        method.start_epoch()

    # accumulate connectivity scores
    for i in xrange(0, n_cons, block_size):
        con_idx = slice(i, i + block_size)
        if adaptive:
            csd = _csd_from_mt(x_mt[idx_map[0][con_idx]],
                               x_mt[idx_map[1][con_idx]],
                               weights[idx_map[0][con_idx]],
                               weights[idx_map[1][con_idx]])
        else:
            csd = _csd_from_mt(x_mt[idx_map[0][con_idx]],
                               x_mt[idx_map[1][con_idx]],
                               weights, weights)

        for method in con_methods:
            method.accumulate(con_idx, csd)

    return con_methods, psd


def _get_n_epochs(epochs, n):
    """Generator that returns lists with at most n epochs"""
    epochs_out = []
    for e in epochs:
        epochs_out.append(e)
        if len(epochs_out) >= n:
            yield epochs_out
            epochs_out = []
    yield epochs_out


# map names to estimator types
CON_METHOD_MAP = {'coh': CohEst, 'cohy': CohyEst, 'imcoh': ImCohEst,
                  'pli': PLIEst, 'wpli': WPLIEst,
                  'wpli2_debias': WPLIDebiasedEst}


@verbose
def spectral_connectivity(data, method='coh', indices=None, sfreq=2 * np.pi,
                          fmin=0, fmax=np.inf, fskip=0, faverage=False,
                          tmin=None, tmax=None, bandwidth=None, adaptive=False,
                          low_bias=True, block_size=1000, n_jobs=1,
                          verbose=None):
    """Compute various frequency-domain connectivity measures

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross- and power spectral
    densities (CSD/PSD) Sxy(f) and Sxx(f), Syy(f), respectively,
    which are computed using a multi-taper method.

    By default, the connectivity between all signals is computed (only
    connections corresponding to the lower-triangular part of the
    connectivity matrix). If one is only interested in the connectivity
    between some signals, the "indices" parameter can be used. For example,
    to compute the connectivity between the signal with index 0 and signals
    "2, 3, 4" (a total of 3 connections) one can use the following:

    indices = (np.array([0, 0, 0],    # row indices
               np.array([2, 3, 4])))  # col indices

    con_flat = spectral_connectivity(data, method='coh', indices=indices, ...)

    In this case con_flat.shape = (3, n_freqs). The connectivity scores are
    in the same order as defined indices.

    Supported Connectivity Measures
    -------------------------------
    The connectivity method(s) is specified using the "method" parameter. The
    following methods are supported (note: E[] denotes average over epochs).
    Multiple measures can be computed at once by using a list/tuple, e.g.
    "['coh', 'pli']" to compute coherence and PLI.

    'coh' : Coherence given by

                     | E[Sxy(f)] |
        C(f) = ---------------------------
               sqrt(E[Sxx(f)] * E[Syy(f)])

    'cohy' : Coherency given by

                       E[Sxy(f)]
        C(f) = ---------------------------
               sqrt(E[Sxx(f)] * E[Syy(f)])

    'imcoh' : Imaginary coherence [1] given by

                      Im(E[Sxy(f)])
        C(f) = --------------------------
               sqrt(E[Sxx(f)] * E[Syy(f)])


    'pli' : Phase Lag Index (PLI) [2] given by

        PLI(f) = |E[sign(Im(Sxy(f)))]|


    'wpli' : Weighted Phase Lag Index (WPLI) [3] given by

                   |E[Im(Sxy(f))]|
        WPLI(f) = ------------------
                   E[|Im(Sxy(f))|]

    'wpli2_debias' : Debiased version of squared WPLI, see [3]

    References:

    [1] Nolte et al. "Identifying true brain interaction from EEG data using
        the imaginary part of coherency" Clinical neurophysiology, vol. 115,
        no. 10, pp. 2292-2307, Oct. 2004.

    [2] Stam et al. "Phase lag index: assessment of functional connectivity
        from multi channel EEG and MEG with diminished bias from common
        sources" Human brain mapping, vol. 28, no. 11, pp. 1178-1193,
        Nov. 2007.

    [3] Vinck et al. "An improved index of phase-synchronization for electro-
        physiological data in the presence of volume-conduction, noise and
        sample-size bias" NeuroImage, vol. 55, no. 4, pp. 1548-1565, Apr. 2011.

    Parameters
    ----------
    data : array, shape=(n_epochs, n_signals, n_times)
           or list/generator of SourceEstimate
           or Epochs
        The data from which to compute connectivity.
    method : (string | object) or a list thereof
        Connectivity measure(s) to compute.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which to compute
        connectivity. If None, all connections are computed.
    sfreq : float
        The sampling frequency.
    fmin : float | tuple of floats
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 20.) for two bands with 8Hz and 20Hz lower freq.
    fmax : float | tuple of floats
        The upper frequency of interest. Multiple bands are dedined using
        a tuple, e.g. (13., 30.) for two band with 13Hz and 30Hz upper freq.
    fskip : int
        Omit every "(fskip + 1)-th" frequency bin to decimate in frequency
        domain.
    faverage : boolean
        Average connectivity scores for each frequency band. If True,
        the output freqs will be a list with arrays of the frequencies
        that were averaged.
    tmin : float | None
        Time to start connectivity estimation. Only supported if data is
        Epochs or a list of SourceEstimate
    tmax : float | None
        Time to end connectivity estimation. Only supported if data is
        Epochs or a list of SourceEstimate.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    block_size : int
        How many connections to compute at once (higher numbers are faster
        but require more memory).
    n_jobs : int
        How many epochs to process in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    con : array | list of arrays
        Computed connectivity measure(s). If "indices" is None, the first
        two dimensions have shape (n_signals, n_signals) otherwise the
        first dimension is len(indices[0]). The remaining dimensions are
        method dependent.
    freqs : array
        Frequency points at which the coherency was computed.
    n_epochs : int
        Number of epochs used for computation.
    n_tapers : int
        The number of DPSS tapers used.
    """

    if n_jobs > 1:
        parallel, my_epoch_spectral_connectivity, _ = \
                parallel_func(_epoch_spectral_connectivity, n_jobs,
                              verbose=verbose)

    # format fmin and fmax and check inputs
    fmin = np.asarray((fmin,)).ravel()
    fmax = np.asarray((fmax,)).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')

    n_bands = len(fmin)

    # assign names to connectivity methods
    if not isinstance(method, (list, tuple)):
        # make it a tuple
        method = [method]

    n_methods = len(method)
    con_method_types = []
    for m in method:
        if m in CON_METHOD_MAP:
            method = CON_METHOD_MAP[m]
            con_method_types.append(method)
        elif isinstance(m, basestring):
            raise ValueError('%s is not a valid connectivity method')
        else:
            # add custom method
            con_method_types.append(m)

    # determine how many arguments the compute_con_function needs
    n_comp_args = [len(getargspec(mtype.compute_con).args)
                   for mtype in con_method_types]

    # we only support 3 or 5 arguments
    if any([n not in (3, 5) for n in n_comp_args]):
        raise ValueError('The compute_con function needs to have either '
                         '3 or 5 arguments')

    # if none of the comp_con functions needs the PSD, we don't estimate it
    accumulate_psd = any([n == 5 for n in n_comp_args])

    # by default we assume time starts at zero
    tmintmax_support = False
    tmin_idx = None
    tmax_idx = None
    tmin_true = None
    tmax_true = None

    if isinstance(data, Epochs):
        tmin_true = data.times[0]
        tmax_true = data.times[-1]
        if tmin is not None:
            tmin_idx = np.argmin(np.abs(data.times - tmin))
            tmin_true = data.times[tmin_idx]
        if tmax is not None:
            tmax_idx = np.argmin(np.abs(data.times - tmax))
            tmax_true = data.times[tmax_idx]
        tmintmax_support = True

    # loop over data; it could be a generator that returns
    # (n_signals x n_times) arrays or SourceEstimates
    epoch_idx = 0
    logger.info('Connectivity computation...')
    for epoch_block in _get_n_epochs(data, n_jobs):

        if epoch_idx == 0:
            first_epoch = epoch_block[0]

            if isinstance(first_epoch, SourceEstimate):
                tmin_true = first_epoch.times[0]
                tmax_true = first_epoch.times[-1]
                if tmin is not None:
                    tmin_idx = np.argmin(np.abs(first_epoch.times - tmin))
                    tmin_true = first_epoch.times[tmin_idx]
                if tmax is not None:
                    tmax_idx = np.argmin(np.abs(first_epoch.times - tmax))
                    tmax_true = first_epoch.times[tmax_idx]
                tmintmax_support = True
                first_epoch = first_epoch.data

            if not tmintmax_support and (tmin is not None or tmax is not None):
                raise ValueError('tmin and tmax are only supported if data is '
                                 'Epochs or a list of SourceEstimate')

            # we want to include the sample at tmax_idx
            tmax_idx = tmax_idx + 1 if tmax_idx is not None else None
            n_signals, n_times = first_epoch[:, tmin_idx:tmax_idx].shape

            # if we are not using Epochs or SourceEstimate, we assume time
            # starts at zero
            if tmin_true is None:
                tmin_true = 0.
            if tmax_true is None:
                tmax_true = n_times / float(sfreq)

            logger.info('    using t=%0.3fs..%0.3fs for estimation (%d points)'
                        % (tmin_true, tmax_true, n_times))

            # compute standardized half-bandwidth
            if bandwidth is not None:
                half_nbw = float(bandwidth) * n_times / (2 * sfreq)
            else:
                half_nbw = 4

            # compute dpss windows
            n_tapers_max = int(2 * half_nbw)
            dpss, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                         low_bias=low_bias)
            n_tapers = len(eigvals)
            logger.info('    using %d DPSS windows' % n_tapers)

            if adaptive and len(eigvals) < 3:
                warn('Not adaptively combining the spectral estimators '
                     'due to a low number of tapers.')
                adaptive = False

            if indices is None:
                # only compute r for lower-triangular region
                indices_use = np.tril_indices(n_signals, -1)
            else:
                indices_use = check_indices(indices)

            # number of connectivities to compute
            n_cons = len(indices_use[0])

            logger.info('    computing connectivity for %d connections'
                        % n_cons)

            # decide which frequencies to keep
            freqs_all = fftfreq(n_times, 1. / sfreq)
            freqs_all = freqs_all[freqs_all >= 0]

            # create a frequency mask for all bands
            freq_mask = np.zeros(len(freqs_all), dtype=np.bool)
            for f_lower, f_upper in zip(fmin, fmax):
                freq_mask |= (freqs_all >= f_lower) & (freqs_all <= f_upper)

            # possibly skip frequency points
            for pos in xrange(fskip):
                freq_mask[pos + 1::fskip + 1] = False

            # the frequency points where we compute connectivity
            freqs = freqs_all[freq_mask]

            # get the freq. indices and points for each band
            freq_idx_bands = [np.where((freqs >= fl) & (freqs <= fu))[0]
                              for fl, fu in zip(fmin, fmax)]
            freqs_bands = [freqs[freq_idx] for freq_idx in freq_idx_bands]

            n_freqs = np.sum(freq_mask)
            if n_bands == 1:
                logger.info('    frequencies: %0.1fHz..%0.1fHz (%d points)'
                            % (freqs_bands[0][0], freqs_bands[0][-1], n_freqs))
            else:
                logger.info('    computing connectivity for the bands:')
                for i, bfreqs in enumerate(freqs_bands):
                    logger.info('     band %d: %0.1fHz..%0.1fHz (%d points)'
                                % (i + 1, bfreqs[0], bfreqs[-1], len(bfreqs)))

            if faverage:
                logger.info('    connectivity scores will be averaged for '
                            'each band')

            # unique signals for which we actually need to compute PSD etc.
            sig_idx = np.unique(np.r_[indices_use[0], indices_use[1]])

            # map indices to unique indices
            idx_map = [np.searchsorted(sig_idx, ind) for ind in indices_use]

            # allocate space to accumulate PSD
            if accumulate_psd:
                psd = np.zeros((len(sig_idx), n_freqs))
            else:
                psd = None

            # create instances of the connectivity methods
            con_methods = [mtype(n_cons, n_freqs)
                           for mtype in con_method_types]

            sep = ', '
            metrics_str = sep.join([method.name for method in con_methods])
            logger.info('    the following metrics will be computed: %s'
                        % metrics_str)

        for i, this_epoch in enumerate(epoch_block):
            if isinstance(this_epoch, SourceEstimate):
                # allow data to be a list of source estimates
                epoch_block[i] = this_epoch.data[:, tmin_idx:tmax_idx]
            else:
                epoch_block[i] = this_epoch[:, tmin_idx:tmax_idx]

        # check dimensions
        for this_epoch in epoch_block:
            if this_epoch.shape != (n_signals, n_times):
                raise ValueError('all epochs must have the same shape')

        if n_jobs == 1:
            # no parallel processing
            for this_epoch in epoch_block:
                if this_epoch.shape != (n_signals, n_times):
                    raise ValueError('all epochs must have the same shape')

                logger.info('    computing connectivity for epoch %d'
                            % (epoch_idx + 1))

                # con methods and psd are updated inplace
                _epoch_spectral_connectivity(this_epoch[sig_idx], sfreq, dpss,
                    eigvals, freq_mask, adaptive, idx_map, block_size, psd,
                    accumulate_psd, con_method_types, con_methods,
                    accumulate_inplace=True)
                epoch_idx += 1
        else:
            # process epochs in parallel
            logger.info('    computing connectivity for epochs %d..%d'
                        % (epoch_idx + 1, epoch_idx + len(epoch_block)))

            out = parallel(my_epoch_spectral_connectivity(this_epoch[sig_idx],
                    sfreq, dpss, eigvals, freq_mask, adaptive, idx_map,
                    block_size, psd, accumulate_psd, con_method_types, None,
                    accumulate_inplace=False) for this_epoch in epoch_block)

            # do the accumulation
            for this_out in out:
                for method, parallel_method in zip(con_methods, this_out[0]):
                    method.combine(parallel_method)
                if accumulate_psd:
                    psd += this_out[1]

            epoch_idx += len(epoch_block)

    # normalize
    n_epochs = epoch_idx + 1
    if accumulate_psd:
        psd /= n_epochs

    # compute final connectivity scores
    con = []
    for method, n_args in zip(con_methods, n_comp_args):
        if n_args == 3:
            # compute all scores at once
            method.compute_con(slice(0, n_cons), n_epochs)
        else:
            # compute scores block-wise to save memory
            for i in xrange(0, n_cons, block_size):
                con_idx = slice(i, i + block_size)
                psd_xx = psd[idx_map[0][con_idx]]
                psd_yy = psd[idx_map[1][con_idx]]
                method.compute_con(con_idx, n_epochs, psd_xx, psd_yy)

        # get the connectivity scores
        this_con = method.con_scores

        if this_con.shape[0] != n_cons:
            raise ValueError('First dimension of connectivity scores must be '
                             'the same as the number of connections')
        if faverage:
            if this_con.shape[1] != n_freqs:
                raise ValueError('2nd dimension of connectivity scores must '
                                 'be the same as the number of frequencies')
            con_shape = (n_cons, n_bands) + this_con.shape[2:]
            this_con_bands = np.empty(con_shape, dtype=this_con.dtype)
            for band_idx in xrange(n_bands):
                this_con_bands[:, band_idx] =\
                    np.mean(this_con[:, freq_idx_bands[band_idx]], axis=1)
            this_con = this_con_bands

        con.append(this_con)

    if indices is None:
        # return all-to-all connectivity matrices
        logger.info('    assembling 3D connectivity matrix')
        con_flat = con
        con = []
        for this_con_flat in con_flat:
            this_con = np.zeros((n_signals, n_signals)
                                + this_con_flat.shape[1:],
                                dtype=this_con_flat.dtype)
            this_con[indices_use] = this_con_flat
            con.append(this_con)

    logger.info('[Connectivity computation done]')

    if n_methods == 1:
        # for a single method return connectivity directly
        con = con[0]

    if faverage:
        # for each band we return the frequencies that were averaged
        freqs = freqs_bands

    return con, freqs, n_epochs, n_tapers
