import numpy as np
from numba import int64, float64, boolean, njit
from numba.experimental import jitclass
from optikon import Propositionalization, max_weighted_support_bb, greedy_maximization, equal_width_propositionalization, full_propositionalization, WeightedSupport, NormalizedWeightedSupport
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

@jitclass
class LeastSquaresRisk:

    def __init__(self):
        pass

    def gradient(self, spec, state):
        return state.phi[:, :state.current_features].dot(state.coef[:state.current_features]) - spec.y
    
    def hessian_diagonal(self, spec, state):
        return np.ones_like(spec.y)
    
    def value(self, spec, state):
        return np.sum((state.phi[:, :state.current_features].dot(state.coef[:state.current_features]) - spec.y))**2

@njit
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@jitclass
class LogisticRisk:

    def __init__(self):
        pass

    def gradient(self, spec, state):
        return sigmoid(state.phi[:, :state.current_features].dot(state.coef[:state.current_features])) - spec.y
    
    def hessian_diagonal(self, spec, state):
        p = sigmoid(state.phi[:, :state.current_features].dot(state.coef[:state.current_features]))
        return p*(1-p)

# @njit
# def gradient_least_squares(spec, state):
#     return state.phi[:, :state.current_features].dot(state.coef[:state.current_features]) - spec.y

# @njit
# def gradient_logistic_loss(spec, state):
#     return sigmoid(state.phi[:, :state.current_features].dot(state.coef[:state.current_features])) - spec.y

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

@jitclass
class BranchAndBoundGradientSumBaseLearner:

    max_depth: int64
    props: Propositionalization

    def __init__(self, spec, max_depth=5, prop_fac=equal_width_propositionalization):
        self.max_depth = max_depth
        self.props = prop_fac(spec.x)

    def compute(self, spec, state, risk_function):
        g = risk_function.gradient(spec, state)

        opt_q_pos, opt_val_pos, _ = max_weighted_support_bb(spec.x, g, self.props, self.max_depth)
        opt_q_neg, opt_val_neg, _ = max_weighted_support_bb(spec.x, -g, self.props, self.max_depth)
        if opt_val_pos >= opt_val_neg:
            return opt_q_pos
        else:
            return opt_q_neg


@jitclass
class GreedyGradientSumBaseLearner:

    max_depth: int64

    def __init__(self, spec, max_depth=5, prop_factory=None):
        self.max_depth = max_depth

    def compute(self, spec, state, risk_function):
        g = risk_function.gradient(spec, state)

        opt_q_pos, opt_val_pos, _ = greedy_maximization(spec.x, WeightedSupport(g), self.max_depth)
        opt_q_neg, opt_val_neg, _ = greedy_maximization(spec.x, WeightedSupport(-g), self.max_depth)
        if opt_val_pos >= opt_val_neg:
            return opt_q_pos
        else:
            return opt_q_neg
        
@jitclass
class GreedyTraditionalGradientBoostingBaseLearner:

    max_depth: int64

    def __init__(self, spec, max_depth=5, prop_factory=None):
        self.max_depth = max_depth

    def compute(self, spec, state, risk_function):
        g = risk_function.gradient(spec, state)

        opt_q_pos, opt_val_pos, _ = greedy_maximization(spec.x, NormalizedWeightedSupport(g, None, 2, 0), self.max_depth)
        opt_q_neg, opt_val_neg, _ = greedy_maximization(spec.x, NormalizedWeightedSupport(-g, None, 2, 0), self.max_depth)
        if opt_val_pos >= opt_val_neg:
            return opt_q_pos
        else:
            return opt_q_neg

@jitclass
class GreedyExtremeGradientBoostingBaseLearner:

    max_depth: int64
    lam: float64

    def __init__(self, spec, max_depth=5, prop_factory=None):
        self.max_depth = max_depth
        self.lam = spec.lam

    def compute(self, spec, state, risk_function):
        g = risk_function.gradient(spec, state)
        h = risk_function.hessian_diagonal(spec, state)

        opt_q_pos, opt_val_pos, _ = greedy_maximization(spec.x, NormalizedWeightedSupport(g, h, 2, self.lam), self.max_depth)
        opt_q_neg, opt_val_neg, _ = greedy_maximization(spec.x, NormalizedWeightedSupport(-g, h, 2, self.lam), self.max_depth)
        if opt_val_pos >= opt_val_neg:
            return opt_q_pos
        else:
            return opt_q_neg

@njit
def gradient_sum_rule_ensemble(spec, state, fit_function, base_learner, risk_function):
    qs = List()
    if spec.intercept:
        qs.append(Propositionalization(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64))) 
        state.phi[:, state.current_features] = 1
        state.current_features += 1
        fit_function(spec, state)
        
    for _ in range(spec.max_features):
        qs.append(base_learner.compute(spec, state, risk_function))

        state.phi[qs[-1].support_all(spec.x), state.current_features] = 1
        state.current_features += 1

        fit_function(spec, state)
    return state.coef, qs

class BaseRuleBoostingEstimator(BaseEstimator):

    prop_options = {
        'equal_width': equal_width_propositionalization,
        'full': full_propositionalization
    } 

    baselearner_options = {
        ('greedy', 'gradient_sum'): GreedyGradientSumBaseLearner,
        ('greedy', 'traditional'): GreedyTraditionalGradientBoostingBaseLearner,
        ('greedy', 'extreme'): GreedyExtremeGradientBoostingBaseLearner,
        ('bb', 'gradient_sum'): BranchAndBoundGradientSumBaseLearner,
    }

    def __init__(self, 
                 spec_factory, 
                 state_factory, 
                 risk_function, 
                 fit_function, num_rules=3, 
                 fit_intercept=True, 
                 lam=0.0, 
                 baselearner='greedy',
                 objective='gradient_sum',
                 max_depth=5,
                 prop='equal_width'):
        self.num_rules = num_rules
        self.fit_intercept = fit_intercept
        self.lam = lam
        self.spec_factory = spec_factory
        self.state_factory = state_factory
        self.risk_function = risk_function
        self.fit_function = fit_function
        self.baselearner = baselearner
        self.objective = objective
        self.max_depth = max_depth
        self.prop = prop

    def fit(self, x, y):
        spec = self.spec_factory(y, x, self.num_rules, self.fit_intercept, self.lam)
        base_learner_function = self.baselearner_options[(self.baselearner, self.objective)](spec, self.max_depth, self.prop_options[self.prop])
        state = self.state_factory(spec)
        self.coef_, self.q_ = gradient_sum_rule_ensemble(spec, state, self.fit_function, base_learner_function, self.risk_function)
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
    Rule-based regressor using corrective gradient boosting.

    Parameters
    ----------
    num_rules : int, default=3
        Maximum number of rules to fit.
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    lam : float, default=1.0
        L2 regularization parameter.
    baselearner : 'greedy' or 'bb', default='greedy'
        Choice of greedy conjunction optimisation or branch-and-bound rule condition optimisation
    max_depth : int, default=4
        Maximum depth of rule condition search (for both bb and greedy).
    prop : 'equal_width' or 'full', default='equal_width'
        Choice for available thresholds for branch-and-bound optimisation (ignored for greedy).

    Examples
    --------
    >>> from ruleboost import RuleBoostingRegressor
    >>> import numpy as np
    >>> x = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]])
    >>> y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
    >>> model = RuleBoostingRegressor(num_rules=2, lam=0.0, fit_intercept=True).fit(x, y)
    >>> print(model.rules_str()) # doctest: +NORMALIZE_WHITESPACE
        +4.000 if  
        -3.000 if x1 <= 0.650 
        -1.000 if x1 <= 0.350 
    >>> np.round(model.predict(x), 3)
    array([0., 0., 0., 1., 1., 1., 4., 4., 4.])

    >>> model2 = RuleBoostingRegressor(num_rules=2, lam=0.0, fit_intercept=True, objective='traditional').fit(x, y)
    >>> print(model2.rules_str()) # doctest: +NORMALIZE_WHITESPACE
    +1.000 if  
    +3.000 if x1 >= 0.650 
    -1.000 if x1 <= 0.350 
    """

    def __init__(self, num_rules=3, fit_intercept=True, lam=1.0, baselearner='greedy', objective='gradient_sum', max_depth=4, prop='equal_width'):
        super().__init__(RegressionSpec, 
                         IncrementalLeastSquaresBoostingState.from_spec, 
                         LeastSquaresRisk(), 
                         fit_minimum_squared_loss_coefs_incrementally, 
                         num_rules, 
                         fit_intercept, 
                         lam, 
                         baselearner,
                         objective,
                         max_depth,
                         prop)

class RuleBoostingClassifier(BaseRuleBoostingEstimator, ClassifierMixin):
    """
    Rule-based classifier using corrective gradient boosting.

    Parameters
    ----------
    num_rules : int, default=3
        Maximum number of rules to fit.
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    lam : float, default=1.0
        L2 regularization parameter.
    baselearner : 'greedy' or 'bb', default='greedy'
        Choice of greedy conjunction optimisation or branch-and-bound rule condition optimisation
    max_depth : int, default=4
        Maximum depth of rule condition search (for both bb and greedy).
    prop : 'equal_width' or 'full', default='equal_width'
        Choice for available thresholds for branch-and-bound optimisation (ignored for greedy).

    Examples
    --------
    >>> from ruleboost import RuleBoostingClassifier
    >>> import numpy as np
    >>> x = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]])
    >>> y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    >>> model = RuleBoostingClassifier(num_rules=1, fit_intercept=True).fit(x, y)
    >>> print(model.rules_str()) # doctest: +NORMALIZE_WHITESPACE
        -0.475 if  
        +0.675 if x1 >= 0.350 & x1 <= 0.650
    >>> model.predict(x)
    array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    >>> np.round(model.predict_proba(x)[:, 1], 2)
    array([0.38, 0.38, 0.38, 0.55, 0.55, 0.55, 0.38, 0.38, 0.38])

    >>> model2 = RuleBoostingClassifier(num_rules=1, fit_intercept=True, objective='extreme', lam=2.0).fit(x, y)
    >>> print(model2.rules_str()) # doctest: +NORMALIZE_WHITESPACE
    -0.284 if  
    +0.361 if x1 >= 0.350 & x1 <= 0.650 
    """

    def __init__(self, num_rules=3, fit_intercept=True, lam=1.0, baselearner='greedy', objective='gradient_sum', max_depth=4, prop='equal_width'):
        super().__init__(ClassificationSpec, 
                         BoostingState.from_spec, 
                         LogisticRisk(), 
                         fit_min_logistic_loss_coefs, 
                         num_rules, 
                         fit_intercept, 
                         lam, 
                         baselearner,
                         objective,
                         max_depth,
                         prop)

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
    doctest.testmod(verbose=True)