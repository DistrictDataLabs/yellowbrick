##########################################################################
## Imports
##########################################################################

import pytest

from yellowbrick.exceptions import NotFitted


##########################################################################
## NotFitted Exception Tests
##########################################################################


class TestExceptions(object):
    """
    Test exception specific code and utilities
    """

    @pytest.mark.parametrize("method", ["transform", None])
    def test_not_fitted_from_estimator(self, method):
        """
        Ensure not fitted can be raised directly from an estimator
        """
        msg = "instance is not fitted yet, please call fit"
        with pytest.raises(NotFitted, match=msg):
            raise NotFitted.from_estimator(self, method)
