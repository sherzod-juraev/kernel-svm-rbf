from numpy import exp, zeros, ndarray, dot


class KernelSVMRBF:
    """Kernel SVM RBF

    """

    def __init__(self, C, gamma: float = .5, /):

        self.C = C
        self.gamma = gamma
        self.a = None
        self.b = 0

    def fit(self, X: ndarray, y: ndarray, /) -> bool:
        """Fit training data

        Parameters
        -----------

        X: [n_sample, m_feature],
            shape={n_sample, m_feature}
        y: [n_sample]
            shape={n_sample]
        """

        n_sample = X.shape[0]
        self.a = zeros(shape=X.shape[0])
        K = zeros(shape=(n_sample, n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                if i != j:
                    K[i][j] = exp(-(self.gamma * ((X[i] - X[j]) ** 2).sum()))
                else:
                    K[i][j] = 1
        f_x = dot(self.a * y, K) + self.b
        E = f_x - y
