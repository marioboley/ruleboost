import numpy as np
from numba import int64, float64, boolean, njit
from numba.experimental import jitclass
from optikon import max_weighted_support, equal_width_propositionalization
from numba.typed import List
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

regression_spec_spec = [
    ('y', float64[:]),
    ('x', float64[:, :]),
    ('max_features', int64),
    ('intercept', boolean),
    ('lam', float64)
]

@jitclass(regression_spec_spec)
class RegressionSpec:
    def __init__(self, y, x, max_features, intercept, lam):
        self.y = y
        self.x = x
        self.max_features = max_features
        self.intercept = intercept
        self.lam = lam

classification_spec_spec = [
    ('y', int64[:]),
    ('x', float64[:, :]),
    ('max_features', int64),
    ('intercept', boolean),
    ('lam', float64),
    ('max_iter', int64),
    ('tol', float64)
]

@jitclass(classification_spec_spec)
class ClassificationSpec:
    def __init__(self, y, x, max_features, intercept, lam):
        self.y = y
        self.x = x
        self.max_features = max_features
        self.intercept = intercept
        self.lam = lam
        self.max_iter=100
        self.tol=1e-6

state_spec = [
    ('phi', float64[:, :]),
    ('coef', float64[:]),
    ('current_features', int64),
]

@jitclass(state_spec)
class BoostingState:
    def __init__(self, phi, coef, current_features):
        self.phi = phi
        self.coef = coef
        self.current_features = current_features

    @staticmethod
    def from_spec(spec):
        phi = np.zeros(shape=(len(spec.y), spec.max_features+spec.intercept))
        coef = np.zeros(spec.max_features+spec.intercept)
        current_features = 0
        return BoostingState(phi, coef, current_features)

incremental_ls_spec = [*state_spec,
    ('gram', float64[:, :]),
    ('chol', float64[:, :]),
]

@jitclass(incremental_ls_spec)
class IncrementalLeastSquaresBoostingState:
    def __init__(self, phi, coef, current_features, gram, chol):
        self.phi = phi
        self.coef = coef
        self.current_features = current_features
        self.gram = gram
        self.chol = chol

    @staticmethod
    def from_spec(spec):
        p = spec.max_features+spec.intercept
        phi = np.zeros(shape=(len(spec.y), p))
        g =  np.zeros((p, p))
        l = np.zeros((p, p))
        coef = np.zeros(p)
        current_features = 0
        return IncrementalLeastSquaresBoostingState(phi, coef, current_features, g, l)

@njit
def gradient_least_squares(spec, state):
    return state.phi[:, :state.current_features].dot(state.coef[:state.current_features]) - spec.y

@njit
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@njit
def gradient_logistic_loss(spec, state):
    return sigmoid(state.phi[:, :state.current_features].dot(state.coef[:state.current_features])) - spec.y

@njit
def fit_minimum_squared_loss_coefs_incrementally(spec, state):
    x, y = state.phi, spec.y
    g, l = state.gram, state.chol
    coef = state.coef[:state.current_features]
    j = state.current_features - 1

    # Update Gramian
    g[j, :j] = x[:, :j].T @ x[:, j]
    g[:j, j] = g[j, :j]
    g[j, j] = x[:, j] @ x[:, j]

    if j!=0 or not spec.intercept:
        g[j, j] += spec.lam

    # Compute RHS
    b = np.zeros(j + 1)
    for i in range(j + 1):
        b[i] = x[:, i] @ y

    # Cholesky update: compute row j of l
    for k in range(j):
        s = 0.0
        for m in range(k):
            s += l[j, m] * l[k, m]
        l[j, k] = (g[j, k] - s) / l[k, k]
    s = 0.0
    for m in range(j):
        s += l[j, m] ** 2
    l[j, j] = np.sqrt(g[j, j] - s)

    # Solve l z = b  (forward solve writing z into coeff)
    for i in range(j + 1):
        s = 0.0
        for k in range(i):
            s += l[i, k] * coef[k]
        coef[i] = (b[i] - s) / l[i, i]

    # Solve l' coef = z  (backward solve, in-place)
    for i in range(j, -1, -1):
        s = 0.0
        for k in range(i + 1, j + 1):
            s += l[k, i] * coef[k]
        coef[i] = (coef[i] - s) / l[i, i]

@njit
def fit_min_logistic_loss_coefs(spec, state):
    phi = state.phi[:, :state.current_features]
    _, d = phi.shape
    beta = state.coef[:d]
    
    for _ in range(spec.max_iter):
        p = sigmoid(phi.dot(beta))
        grad = phi.T @ (p - spec.y) + 2 * spec.lam * beta
        s = p * (1 - p)
        h = phi.T @ (phi * s[:, None]) + 2 * spec.lam * np.eye(d)
        delta = np.linalg.solve(h, grad)
        beta -= delta
        if np.linalg.norm(delta) < spec.tol:
            break

@njit
def gradient_sum_rule_ensemble(spec, state, props, fit_function, gradient_function, max_depth=5):
    qs = List()
    if spec.intercept:
        qs.append(props[0:0]) 
        state.phi[:, state.current_features] = 1
        state.current_features += 1
        fit_function(spec, state)
        
    for _ in range(spec.max_features):
        g = gradient_function(spec, state)

        opt_key_pos, opt_val_pos, _, _ = max_weighted_support(spec.x, g, props, max_depth)
        opt_key_neg, opt_val_neg, _, _ = max_weighted_support(spec.x, -g, props, max_depth)
        if opt_val_pos >= opt_val_neg:
            qs.append(props[opt_key_pos])
        else:
            qs.append(props[opt_key_neg])

        state.phi[qs[-1].support_all(spec.x), state.current_features] = 1
        state.current_features += 1

        fit_function(spec, state)
    return state.coef, qs


class BaseRuleBoostingEstimator(BaseEstimator):

    def __init__(self, 
                 spec_factory, 
                 state_factory, 
                 gradient_function, 
                 fit_function, num_rules=3, 
                 fit_intercept=True, 
                 lam=0.0, 
                 prop_factory=equal_width_propositionalization,
                 max_depth=5):
        self.num_rules = num_rules
        self.fit_intercept = fit_intercept
        self.prop_factory = prop_factory
        self.max_depth = max_depth
        self.lam = lam
        self.spec_factory = spec_factory
        self.state_factory = state_factory
        self.gradient_function = gradient_function
        self.fit_function = fit_function

    def fit(self, x, y):
        props = self.prop_factory(x)
        spec = self.spec_factory(y, x, self.num_rules, self.fit_intercept, self.lam)
        state = self.state_factory(spec)
        self.coef_, self.q_ = gradient_sum_rule_ensemble(spec, state, props, self.fit_function, self.gradient_function)
        return self
    
    def predict(self, x):
        q_matrix = self.transform(x)
        return q_matrix.dot(self.coef_)

    def transform(self, x):
        n = len(x)
        q_matrix = np.zeros(shape=(n, len(self.q_)))
        for i in range(len(self.q_)):
            q_matrix[self.q_[i].support_all(x), i] = 1
        return q_matrix
    
    def rules_str(self):
        res = ''
        for i in range(len(self.q_)):
            res += f'{self.coef_[i]:+.3f} if {self.q_[i].str_from_conj(np.arange(len(self.q_[i])))} {'\n' if i<len(self.q_)-1 else ''}'
        return res

class RuleBoostingRegressor(BaseRuleBoostingEstimator, RegressorMixin):
    """
    Rule-based regressor using gradient boosting with branch-and-bound search for conjunctive condition.

    Parameters
    ----------
    num_rules : int, default=3
        Maximum number of rules to fit.
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    lam : float, default=1.0
        L2 regularization parameter.
    max_depth : int, default=5
        Maximum depth of rule condition tree search.

    Examples
    --------
    >>> from ruleboost import RuleBoostingRegressor
    >>> from optikon import full_propositionalization
    >>> import numpy as np
    >>> x = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]])
    >>> y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
    >>> model = RuleBoostingRegressor(num_rules=2, lam=0.0, fit_intercept=True, prop=full_propositionalization).fit(x, y)
    >>> print(model.rules_str()) # doctest: +NORMALIZE_WHITESPACE
        +4.000 if  
        -3.000 if x1 <= 0.600 
        -1.000 if x1 <= 0.300 
    >>> np.round(model.predict(x), 3)
    array([0., 0., 0., 1., 1., 1., 4., 4., 4.])
    """

    def __init__(self, num_rules=3, fit_intercept=True, lam=1.0, prop=equal_width_propositionalization, max_depth=5):
        super().__init__(RegressionSpec, 
                         IncrementalLeastSquaresBoostingState.from_spec, 
                         gradient_least_squares, 
                         fit_minimum_squared_loss_coefs_incrementally, 
                         num_rules, 
                         fit_intercept, 
                         lam, 
                         prop,
                         max_depth)

class RuleBoostingClassifier(BaseRuleBoostingEstimator, ClassifierMixin):
    """
    Rule-based regressor using gradient boosting with branch-and-bound search for conjunctive condition.

    Parameters
    ----------
    num_rules : int, default=3
        Maximum number of rules to fit.
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    lam : float, default=1.0
        L2 regularization parameter.
    max_depth : int, default=5
        Maximum depth of rule condition tree search.

    Examples
    --------
    >>> from ruleboost import RuleBoostingClassifier
    >>> from optikon import full_propositionalization
    >>> import numpy as np
    >>> x = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]])
    >>> y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    >>> model = RuleBoostingClassifier(num_rules=1, fit_intercept=True, prop=full_propositionalization).fit(x, y)
    >>> print(model.rules_str()) # doctest: +NORMALIZE_WHITESPACE
        -0.475 if  
        +0.675 if x1 >= 0.400 & x1 <= 0.600
    >>> model.predict(x)
    array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    >>> np.round(model.predict_proba(x)[:, 1], 2)
    array([0.38, 0.38, 0.38, 0.55, 0.55, 0.55, 0.38, 0.38, 0.38])
    """

    def __init__(self, num_rules=3, fit_intercept=True, lam=1.0, prop=equal_width_propositionalization, max_depth=5):
        super().__init__(ClassificationSpec, 
                         BoostingState.from_spec, 
                         gradient_logistic_loss, 
                         fit_min_logistic_loss_coefs, 
                         num_rules, 
                         fit_intercept, 
                         lam, 
                         prop,
                         max_depth)

    def fit(self, x, y):
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        return super().fit(x, y_encoded)

    def predict_proba(self, x):
        res = np.zeros((len(x), len(self.classes_)))
        res[:, 1] = sigmoid(super().predict(x))
        res[:, 0] = 1 - res[:, 1]
        return res
    
    def predict(self, x):
        return self.classes_[(super().predict(x)>=0.0).astype(np.int64)]


if __name__=='__main__':
    import doctest
    doctest.testmod()