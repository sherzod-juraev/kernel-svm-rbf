from numpy import exp, zeros, ndarray


class KernelSVMRBF:
    """Kernel SVM RBF

    """

    def __init__(self, C, gamma: float = .5, /):

        self.C = C
        self.gamma = gamma
        self.a = None

    def initialize_alfa(self, length, /):
        """Initialize alfa"""

        self.a = zeros(shape=length)

    def fit(self, X: ndarray, y: ndarray, /) -> bool:
        """Fit training data

        Parameters
        -----------

        X: [n_sample, m_feature],
            shape={n_sample, m_feature}
        y: [n_sample]
            shape={n_sample]
        """

        self.initialize_alfa(X.shape[0])
        K = zeros(shape=(X.shape[0], X.shape[0]))
