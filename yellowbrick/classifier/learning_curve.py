# yellowbrick.regressor.alphas
# Implements alpha selection visualizers for regularization
#
# Author:   Jason Keung <jason.s.keung@gmail.com>
# Created:  Mon May 22 09:22:07 2017 -0500
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: learning_curve.py [] jason.s.keung@gmail.com $


class LearningCurveVisualizer(ModelVisualizer):
    """
    
    """
    def __init__(self, model, train_sizes=None, cv=None, colormap=None, color=None, n_jobs=1, **kwargs):
        
        super(LearningCurveVisualizer, self).__init__(model, **kwargs)
        
        self.colormap = colormap
        self.color = color
        self.cv = cv
        self.n_jobs = n_jobs
        self.train_sizes = train_sizes or np.linspace(.1, 1.0, 5)
        
        # to be set later
        self.train_scores = None
        self.test_scores = None
        self.train_scores_mean = None
        self.train_scores_std = None
        self.test_scores_mean = None
        self.test_scores_std = None
        
    def fit(self, X, y, **kwargs):
        # Run the sklearn learning curve functions with the params
        self.train_sizes, self.train_scores, self.test_scores = learning_curve(
                estimator, X, y, cv=cv, n_jobs=self.n_jobs, train_sizes=self.train_sizes)

        self.train_scores_mean = np.mean(self.train_scores, axis=1)
        self.train_scores_std = np.std(self.train_scores, axis=1)
        self.test_scores_mean = np.mean(self.test_scores, axis=1)
        self.test_scores_std = np.std(self.test_scores, axis=1)
        
        self.draw(**kwargs)
        return self
        
    def draw(self, *kwargs):
        # Add data to the plot
        self.ax.fill_between(self.train_sizes, self.train_scores_mean - self.train_scores_std,
                         self.train_scores_mean + self.train_scores_std, alpha=0.1,
                         color=train_color)

        self.ax.fill_between(self.train_sizes, self.test_scores_mean - self.test_scores_std,
                         self.test_scores_mean + self.test_scores_std, alpha=0.1, color=cv_color)

        self.ax.plot(self.train_sizes, self.train_scores_mean, 'o-', color='b',
                 label="Training score")

        self.ax.plot(self.train_sizes, self.test_scores_mean, 'o-', color='g',
                 label="Cross-validation score")        

    def finalize(self):
        # Set title
        self.set_title('Learning Curve for {}'.format(self.name))
        self.ax.legend(('Training Score', 'Cross-validation Score'), frameon=True, loc='best')
        self.ax.set_xlabel('Training Score')
        self.ax.set_ylabel('Cross-validation Score')
        