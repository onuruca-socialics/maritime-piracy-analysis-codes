import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_corrected_stat(x, y):  # TODO: Refactor
    """calculate Cramers V statistic for categorial-categorial association.
    """
    conf_matrix = pd.crosstab(x, y)

    correct = True
    if conf_matrix.shape[0] == 2:
        correct = False

    chi2, *_ = chi2_contingency(conf_matrix, correction=correct)

    n = sum(conf_matrix.sum())
    phi2 = chi2 / n
    r, k = conf_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    result = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return round(result, 6)
