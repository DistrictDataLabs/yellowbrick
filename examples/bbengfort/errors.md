### Error 1 

Trying to import Yellowbrick:

```

from yellowbrick.text import FreqDistVisualizer
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
<ipython-input-4-7d5d783f521e> in <module>()
----> 1 from yellowbrick.text import FreqDistVisualizer

ImportError: No module named 'yellowbrick'
```

Was not in a virtualenv, tried to get a virtualenv set up; still having problems with yellowbrick. 
Solved: sys.path.append

### Error 2 

On import, deprecation warning:

```
/usr/local/lib/python3.5/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
```

In my user testing I'm using Scikit-Learn 0.18 

### Error 3 

Running `FreqDistVisualizer` in a pipeline:

```
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer 
from yellowbrick.text import FreqDistVisualizer

visualizer = Pipeline([
    ('norm', TextNormalizer()),
    ('count', CountVectorizer(tokenizer=lambda x: x, preprocessor=None, lowercase=False)),
    ('viz', FreqDistVisualizer())
])

visualizer.fit_transform(documents(), labels())
visualizer.named_steps['viz'].poof()
```

I get the following error:

```
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-53-3514380b0c82> in <module>()
      9 ])
     10 
---> 11 visualizer.fit_transform(documents(), labels())
     12 visualizer.named_steps['viz'].poof()

/usr/local/lib/python3.5/site-packages/sklearn/pipeline.py in fit_transform(self, X, y, **fit_params)
    301         Xt, fit_params = self._fit(X, y, **fit_params)
    302         if hasattr(last_step, 'fit_transform'):
--> 303             return last_step.fit_transform(Xt, y, **fit_params)
    304         elif last_step is None:
    305             return Xt

/usr/local/lib/python3.5/site-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    495         else:
    496             # fit method of arity 2 (supervised transformation)
--> 497             return self.fit(X, y, **fit_params).transform(X)
    498 
    499 

AttributeError: 'NoneType' object has no attribute 'transform'
```

This is because the `fit()` method needs to return self. 
Also, the features passed into fit do not match the vector y that I'm passing to it. 
Instead, features needs to be added as input to the initializer, or passed to 
`inverse_transform` as needed on the vectorizer. 

### Error 4 

If I don't call poof() then I get a graphic displayed in the notebook that's not finalized. I think this is going to be a weirdness of `%matplotlib inline` 

![No poof called, no finalize](figures/nopoofnofinal.png)
