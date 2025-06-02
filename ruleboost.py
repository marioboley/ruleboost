import numpy as np
from optikon import max_weighted_support, equal_width_propositionalization

class RuleBoostingRegressor:

    def __init__(self, num_rules=3, max_depth=5):
        self.num_rules = num_rules
        self.max_depth = max_depth

    def fit(self, x, y):
        n = len(x)
        props = equal_width_propositionalization(x)
        self.q_ = []

        q_matrix = np.zeros(shape=(n, self.num_rules))
        self.coef_ = np.zeros(shape=0)
        for i in range(self.num_rules):
            # this is a bit shaky in the first iteration but seems to work as intended
            # should get more robust once one adds a background rule
            y_hat = q_matrix[:, :i].dot(self.coef_) 
            g = y - y_hat

            opt_key_pos, opt_val_pos, _, _ = max_weighted_support(x, g, props, self.max_depth)
            opt_key_neg, opt_val_neg, _, _ = max_weighted_support(x, -g, props, self.max_depth)
            if opt_val_pos >= opt_val_neg:
                self.q_.append(props[opt_key_pos])
            else:
                self.q_.append(props[opt_key_neg])

            q_matrix[self.q_[i].support_all(x), i] = 1
            self.coef_ = np.linalg.solve(q_matrix[:, :i+1].T.dot(q_matrix[:, :i+1]), q_matrix[:, :i+1].T.dot(y))

        return self
    
    def predict(self, x):
        n = len(x)
        q_matrix = np.zeros(shape=(n, len(self.q_)))
        for i in range(len(self.q_)):
            q_matrix[self.q_[i].support_all(x), i] = 1
        return q_matrix.dot(self.coef_)
    
    def __str__(self):
        res = ''
        for i in range(len(self.q_)):
            res += f'{self.coef_[i]:+.3f} if {self.q_[i].str_from_conj(np.arange(len(self.q_[i])))} \n'
        return res
    
    def __repr__(self):
        return f'RuleBoostingRegressor({self.num_rules}, {self.max_depth})'

if __name__=='__main__':
    import doctest
    doctest.testmod()