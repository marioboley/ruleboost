# ruleboost

Learn additive rule ensembles via gradient boosting.

## Usage

```python
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
```
