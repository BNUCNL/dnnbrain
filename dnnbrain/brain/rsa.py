import numpy as np
from scipy.stats import spearmanr, kendalltau
from numpy.typing import NDArray
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

class RSA:
    """
    A class to perform Representational Similarity Analysis (RSA) between Representational Dissimilarity Matrices (RDMs).

    Parameters
    ----------
    rdm1 : NDArray
        The first RDM or a set of RDMs.
    rdm2 : NDArray
        The second RDM or a set of RDMs.
    input_type : str
        Type of input RDMs. One of 'n2n', 'one2n', 'one2one'.
    n_jobs : int
        The number of parallel jobs to run.
    n_iter : int
        The number of iterations for bootstrapping or permutation testing.
    """

    def __init__(
        self,
        rdm1: NDArray,
        rdm2: NDArray,
        input_type: str,
        n_jobs: int,
        n_iter: int,
    ) -> None:
        self._check_input(rdm1, rdm2, input_type)

        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.input_type = input_type
        self.n_jobs = n_jobs
        self.n_iter = n_iter

    def corr_rdm(
        self,
        corr_method: str,
        sig_method: str,
    ) -> tuple[float, NDArray]:
        """
        Compute the correlation between two RDMs with statistical significance testing.

        Parameters
        ----------
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        sig_method : str
            The significance testing method. One of 'bootstrap', 'permutation'.

        Returns
        -------
        corr : float
            The correlation coefficient between the two RDMs.
        stats : NDArray
            The bootstrapped or permuted correlation coefficients, depending on `sig_method`.
        """
        if sig_method == 'bootstrap':
            corr, boot_cs = self.corr_bootstrap(
                self.rdm1, self.rdm2, corr_method, n_bootstraps=self.n_iter, n_jobs=self.n_jobs
            )
            return corr, boot_cs
        elif sig_method == 'permutation':
            corr, perm_cs = self.corr_permutation(
                self.rdm1, self.rdm2, corr_method, n_permutations=self.n_iter, n_jobs=self.n_jobs
            )
            return corr, perm_cs
        else:
            raise ValueError('sig_method must be one of "bootstrap", "permutation"')

    def corr_rdms(
        self,
        corr_method: str,
        sig_method: str,
        alpha: float = 0.05,
    ) -> tuple[list[float], list]:
        """
        Compute the correlations between pairs of RDMs with statistical significance testing.

        Parameters
        ----------
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        sig_method : str
            The significance testing method. One of 'bootstrap', 'permutation'.
        alpha : float, optional
            Significance level for confidence intervals (only used when `sig_method` is 'bootstrap'). Default is 0.05.

        Returns
        -------
        coors : list of float
            List of correlation coefficients between paired RDMs.
        ci_or_p : list
            List of confidence intervals (if `sig_method` is 'bootstrap') or p-values (if `sig_method` is 'permutation').
        """
        n_pair = self.rdm1.shape[0]
        coors = []
        ci = []  # Confidence intervals
        ps = []  # P-values

        for i, (rdm1_i, rdm2_i) in enumerate(zip(self.rdm1, self.rdm2)):
            desc = f'{i + 1}/{n_pair}'

            if sig_method == 'bootstrap':
                corr, boot_cs = self.corr_bootstrap(
                    rdm1_i, rdm2_i,
                    corr_method,
                    n_bootstraps=self.n_iter,
                    n_jobs=self.n_jobs,
                    desc=desc
                )
                ci_lower = np.percentile(boot_cs, 100 * alpha / 2)
                ci_upper = np.percentile(boot_cs, 100 * (1 - alpha / 2))
                ci.append((ci_lower, ci_upper))
                coors.append(corr)

            elif sig_method == 'permutation':
                corr, perm_cs = self.corr_permutation(
                    rdm1_i, rdm2_i, corr_method, n_permutations=self.n_iter, n_jobs=self.n_jobs,
                    desc=desc
                )
                p = np.sum(perm_cs >= corr) / len(perm_cs)
                coors.append(corr)
                ps.append(p)
            else:
                raise ValueError('sig_method must be one of "bootstrap", "permutation"')

        if sig_method == 'bootstrap':
            return coors, ci
        elif sig_method == 'permutation':
            return coors, ps

    def corr_rdm_rdms(
        self,
        corr_method: str,
        sig_method: str,
        alpha: float = 0.05,
    ) -> tuple[list[float], list]:
        """
        Compute the correlations between a single RDM and a set of RDMs with statistical significance testing.

        Parameters
        ----------
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        sig_method : str
            The significance testing method. One of 'bootstrap', 'permutation'.
        alpha : float, optional
            Significance level for confidence intervals (only used when `sig_method` is 'bootstrap'). Default is 0.05.

        Returns
        -------
        corrs : list of float
            List of correlation coefficients between the single RDM and each RDM in the set.
        ci_or_p : list
            List of confidence intervals (if `sig_method` is 'bootstrap') or p-values (if `sig_method` is 'permutation').
        """
        rdm1 = self.rdm1
        rdms = self.rdm2

        corrs = []
        ci = []
        ps = []

        for i, rdm in enumerate(rdms):
            desc = f'{i + 1}/{len(rdms)}'

            if sig_method == 'bootstrap':
                corr, boot_cs = self.corr_bootstrap(
                    rdm1, rdm,
                    corr_method,
                    n_bootstraps=self.n_iter,
                    n_jobs=self.n_jobs,
                    desc=desc
                )
                ci_lower = np.percentile(boot_cs, 100 * alpha / 2)
                ci_upper = np.percentile(boot_cs, 100 * (1 - alpha / 2))
                corrs.append(corr)
                ci.append((ci_lower, ci_upper))

            elif sig_method == 'permutation':
                corr, perm_cs = self.corr_permutation(
                    rdm1, rdm,
                    corr_method,
                    n_permutations=self.n_iter,
                    n_jobs=self.n_jobs,
                    desc=desc
                )
                p = np.sum(perm_cs >= corr) / len(perm_cs)
                corrs.append(corr)
                ps.append(p)
            else:
                raise ValueError('sig_method must be one of "bootstrap", "permutation"')

        if sig_method == 'bootstrap':
            return corrs, ci
        elif sig_method == 'permutation':
            return corrs, ps

    def corr_bootstrap(
        self,
        rdm1: NDArray,
        rdm2: NDArray,
        corr_method: str,
        n_bootstraps: int,
        n_jobs: int,
        **kwargs,
    ) -> tuple[float, NDArray]:
        """
        Compute the correlation between two RDMs and the bootstrapped confidence intervals.

        Parameters
        ----------
        rdm1 : NDArray
            The first RDM.
        rdm2 : NDArray
            The second RDM.
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        n_bootstraps : int
            The number of bootstrap samples.
        n_jobs : int
            The number of parallel jobs to run.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        corr : float
            The correlation coefficient between the two RDMs.
        boot_cs : NDArray
            The array of bootstrapped correlation coefficients.
        """
        v1 = self._rdm2vec(rdm1)
        v2 = self._rdm2vec(rdm2)
        corr = self._corr(v1, v2, corr_method)

        desc = kwargs.get('desc', "Bootstrap process:")
        with tqdm_joblib(total=n_bootstraps, desc=desc, leave=False):
            boot_cs = Parallel(n_jobs=n_jobs)(
                delayed(self._bootstrap)(v1, v2, corr_method) for _ in range(n_bootstraps)
            )

        return corr, np.array(boot_cs)

    def corr_permutation(
        self,
        rdm1: NDArray,
        rdm2: NDArray,
        corr_method: str,
        n_permutations: int,
        n_jobs: int,
        **kwargs,
    ) -> tuple[float, NDArray]:
        """
        Compute the correlation between two RDMs and the distribution of permuted correlations.

        Parameters
        ----------
        rdm1 : NDArray
            The first RDM.
        rdm2 : NDArray
            The second RDM.
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        n_permutations : int
            The number of permutations.
        n_jobs : int
            The number of parallel jobs to run.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        corr : float
            The correlation coefficient between the two RDMs.
        perm_cs : NDArray
            The array of permuted correlation coefficients.
        """
        v1 = self._rdm2vec(rdm1)
        v2 = self._rdm2vec(rdm2)
        corr = self._corr(v1, v2, corr_method)

        desc = kwargs.get('desc', "Permutation process:")
        with tqdm_joblib(total=n_permutations, desc=desc, leave=False):
            perm_cs = Parallel(n_jobs=n_jobs)(
                delayed(self._permutation)(v1, v2, corr_method) for _ in range(n_permutations)
            )

        return corr, np.array(perm_cs)

    def _corr(
        self,
        v1: NDArray,
        v2: NDArray,
        method: str,
    ) -> float:
        """
        Compute the correlation between two vectors.

        Parameters
        ----------
        v1 : NDArray
            The first vector.
        v2 : NDArray
            The second vector.
        method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.

        Returns
        -------
        corr : float
            The correlation coefficient.
        """
        if method == 'spearman':
            corr, _ = spearmanr(v1, v2)
        elif method == 'pearson':
            corr = np.corrcoef(v1, v2)[0, 1]
        elif method == 'kendall':
            corr, _ = kendalltau(v1, v2)
        else:
            raise ValueError('method must be one of "spearman", "pearson", "kendall"')
        return corr

    def _bootstrap(self, v1: NDArray, v2: NDArray, method: str) -> float:
        """
        Perform one bootstrap iteration to compute correlation.

        Parameters
        ----------
        v1 : NDArray
            The first vector.
        v2 : NDArray
            The second vector.
        method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.

        Returns
        -------
        corr : float
            The correlation coefficient for the bootstrap sample.
        """
        n = len(v1)
        indices = np.random.choice(n, n, replace=True)
        v1_boot = v1[indices]
        v2_boot = v2[indices]
        corr = self._corr(v1_boot, v2_boot, method)
        return corr

    def _permutation(self, v1: NDArray, v2: NDArray, method: str) -> float:
        """
        Perform one permutation iteration to compute correlation.

        Parameters
        ----------
        v1 : NDArray
            The first vector.
        v2 : NDArray
            The second vector.
        method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.

        Returns
        -------
        corr : float
            The correlation coefficient for the permuted data.
        """
        v2_perm = np.random.permutation(v2)
        corr = self._corr(v1, v2_perm, method)
        return corr

    def _rdm2vec(
        self,
        rdm: NDArray
    ) -> NDArray:
        """
        Convert a square RDM matrix to a vector of its lower triangle elements.

        Parameters
        ----------
        rdm : NDArray
            The RDM matrix.

        Returns
        -------
        vec : NDArray
            The vectorized lower triangle of the RDM.
        """
        lower_triangle = rdm[np.tril_indices(rdm.shape[0], k=-1)]
        return lower_triangle

    def _check_input(
        self,
        rdm1: NDArray,
        rdm2: NDArray,
        input_type: str,
    ) -> None:
        """
        Validate input RDMs.

        Parameters
        ----------
        rdm1 : NDArray
            The first RDM or a set of RDMs.
        rdm2 : NDArray
            The second RDM or a set of RDMs.
        input_type : str
            Type of input RDMs. One of 'n2n', 'one2n', 'one2one'.

        Raises
        ------
        TypeError
            If inputs are not numpy arrays.
        ValueError
            If `input_type` is invalid or RDM dimensions do not match the expected dimensions for the given `input_type`.
        """
        if not isinstance(rdm1, np.ndarray) or not isinstance(rdm2, np.ndarray):
            raise TypeError(
                f'Input RDMs should be numpy arrays, got {type(rdm1)} and {type(rdm2)}'
            )
        if input_type == 'n2n':
            if rdm1.ndim != 3 or rdm2.ndim != 3:
                raise ValueError('Both RDMs should be 3D arrays for input_type "n2n"')
        elif input_type == 'one2n':
            if rdm1.ndim != 2 or rdm2.ndim != 3:
                raise ValueError(
                    'rdm1 should be a 2D array and rdm2 should be a 3D array for input_type "one2n"'
                )
        elif input_type == 'one2one':
            if rdm1.ndim != 2 or rdm2.ndim != 2:
                raise ValueError('Both RDMs should be 2D arrays for input_type "one2one"')
        else:
            raise ValueError('input_type must be one of "n2n", "one2n", "one2one"')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    def create_linear_patterned_rdm(size):
        """
        Creates an RDM where dissimilarity increases linearly with index difference.
        """
        rdm = np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
        return rdm

    def create_inverse_linear_patterned_rdm(size):
        """
        Creates an RDM where dissimilarity decreases with index difference.
        """
        max_distance = size - 1
        rdm = max_distance - np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
        return rdm

    def create_noisy_rdm(base_rdm, noise_level):
        """
        Adds Gaussian noise to a base RDM.
        """
        noise = np.random.normal(0, noise_level, base_rdm.shape)
        rdm_noisy = base_rdm + noise
        # Ensure the RDM is symmetric and zeros on diagonal
        rdm_noisy = (rdm_noisy + rdm_noisy.T) / 2
        np.fill_diagonal(rdm_noisy, 0)
        return rdm_noisy

    def create_random_rdm(size):
        """
        Creates a random symmetric RDM with zeros on the diagonal.
        """
        random_matrix = np.random.rand(size, size)
        rdm_random = (random_matrix + random_matrix.T) / 2
        np.fill_diagonal(rdm_random, 0)
        return rdm_random

    rdm_one2one_1 = create_linear_patterned_rdm(10)
    rdm_one2one_2 = create_noisy_rdm(rdm_one2one_1, noise_level=0.5)

    rdm_one2n_1 = create_linear_patterned_rdm(10)
    rdm_one2n_n = []

    for _ in range(5):
        rdm_similar = create_noisy_rdm(rdm_one2n_1, noise_level=0.5)
        rdm_one2n_n.append(rdm_similar)
        
    for _ in range(5):
        rdm_random = create_random_rdm(10)
        rdm_one2n_n.append(rdm_random)

    inverse_pattern = create_inverse_linear_patterned_rdm(10)
    for _ in range(5):
        rdm_different = create_noisy_rdm(inverse_pattern, noise_level=0.5)
        rdm_one2n_n.append(rdm_different)

    rdm_one2n_n = np.array(rdm_one2n_n)

    # Generate pairs of RDMs
    rdms_n2n_1 = []
    rdms_n2n_2 = []

    # 5 pairs of similar RDMs
    for _ in range(5):
        base_rdm = create_linear_patterned_rdm(10)
        rdm1 = create_noisy_rdm(base_rdm, noise_level=0.5)
        rdm2 = create_noisy_rdm(base_rdm, noise_level=0.5)
        rdms_n2n_1.append(rdm1)
        rdms_n2n_2.append(rdm2)

    # 5 pairs of random RDMs
    for _ in range(5):
        rdm1 = create_random_rdm(10)
        rdm2 = create_random_rdm(10)
        rdms_n2n_1.append(rdm1)
        rdms_n2n_2.append(rdm2)

    # 5 pairs of different RDMs (one patterned, one inverse patterned)
    for _ in range(5):
        rdm1 = create_noisy_rdm(create_linear_patterned_rdm(10), noise_level=0.5)
        rdm2 = create_noisy_rdm(create_inverse_linear_patterned_rdm(10), noise_level=0.5)
        rdms_n2n_1.append(rdm1)
        rdms_n2n_2.append(rdm2)

    rdms_n2n_1 = np.array(rdms_n2n_1)
    rdms_n2n_2 = np.array(rdms_n2n_2)


    # One2One
    rsa_one2one = RSA(
        rdm1=rdm_one2one_1,
        rdm2=rdm_one2one_2,
        input_type='one2one',
        n_jobs=-1,
        n_iter=200 # easy for testing
    )

    # One2N
    rsa_one2n = RSA(
        rdm1=rdm_one2n_1,
        rdm2=rdm_one2n_n,
        input_type='one2n',
        n_jobs=-1,
        n_iter= 200 # easy for testing
    )

    # N2N
    rsa_n2n = RSA(
        rdm1=rdms_n2n_1,
        rdm2=rdms_n2n_2,
        input_type='n2n',
        n_jobs=-1,
        n_iter= 200
    )

    # Compute correlation and bootstrap confidence interval
    corr_one2one, boot_cs_one2one = rsa_one2one.corr_rdm(
        corr_method='pearson',
        sig_method='bootstrap'
    )

    # Compute 95% confidence interval
    ci_lower_one2one = np.percentile(boot_cs_one2one, 2.5)
    ci_upper_one2one = np.percentile(boot_cs_one2one, 97.5)


    # Compute correlations and p-values
    corrs_one2n, ps_one2n = rsa_one2n.corr_rdm_rdms(
        corr_method='spearman',
        sig_method='permutation'
    )


    # Compute correlations and bootstrap confidence intervals
    coors_n2n, ci_n2n = rsa_n2n.corr_rdms(
        corr_method='kendall',
        sig_method='bootstrap'
    )


    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot for One2One
    axes[0].hist(boot_cs_one2one, bins=30, color='skyblue', edgecolor='black')
    axes[0].axvline(corr_one2one, color='red', linestyle='dashed', linewidth=2, label=f'Observed Corr = {corr_one2one:.4f}')
    axes[0].axvline(ci_lower_one2one, color='green', linestyle='dashed', linewidth=2, label=f'95% CI Lower = {ci_lower_one2one:.4f}')
    axes[0].axvline(ci_upper_one2one, color='green', linestyle='dashed', linewidth=2, label=f'95% CI Upper = {ci_upper_one2one:.4f}')
    axes[0].set_title('One2One: Bootstrapped Correlation Coefficients')
    axes[0].set_xlabel('Correlation Coefficient')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot for One2N
    indices = np.arange(len(corrs_one2n)) + 1
    axes[1].bar(indices, corrs_one2n, color='gray', edgecolor='black')
    axes[1].set_xlabel('RDM Index')
    axes[1].set_ylabel('Correlation Coefficient')
    axes[1].set_title('One2N: Correlation Coefficients')

    for idx in range(len(corrs_one2n)):
        if idx < 5:
            axes[1].bar(idx + 1, corrs_one2n[idx], color='green')
        elif idx < 10:
            axes[1].bar(idx + 1, corrs_one2n[idx], color='blue')
        else:
            axes[1].bar(idx + 1, corrs_one2n[idx], color='red')

    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Similar RDMs'),
        Patch(facecolor='blue', edgecolor='black', label='Random RDMs'),
        Patch(facecolor='red', edgecolor='black', label='Different RDMs')
    ]
    axes[1].legend(handles=legend_elements)

    # Plot for N2N
    indices_n2n = np.arange(len(coors_n2n)) + 1
    corrs_array_n2n = np.array(coors_n2n)
    ci_lower_array_n2n = np.array([ci_pair[0] for ci_pair in ci_n2n])
    ci_upper_array_n2n = np.array([ci_pair[1] for ci_pair in ci_n2n])

    axes[2].errorbar(indices_n2n, corrs_array_n2n,
                    yerr=[corrs_array_n2n - ci_lower_array_n2n, ci_upper_array_n2n - corrs_array_n2n],
                    fmt='o', ecolor='darkblue', capsize=5, color='blue')

    axes[2].set_xlabel('Pair Index')
    axes[2].set_ylabel('Correlation Coefficient')
    axes[2].set_title('N2N: Correlation Coefficients with 95% CI')

    for idx in range(len(coors_n2n)):
        if idx < 5:
            axes[2].plot(idx + 1, coors_n2n[idx], 'go')
        elif idx < 10:
            axes[2].plot(idx + 1, coors_n2n[idx], 'bo')
        else:
            axes[2].plot(idx + 1, coors_n2n[idx], 'ro')

    axes[2].legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()
#%%
