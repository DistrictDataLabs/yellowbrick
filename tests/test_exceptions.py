##########################################################################
## Imports
##########################################################################
import pytest
from yellowbrick.exceptions import NotFitted

##########################################################################
## NotFitted Exception Tests
##########################################################################
class TestNotFitted(object):
    """
    Test Not Fitted Error
    """    
    def test_correct_message(self):
        "Ensure that NotFitted error is raised properly"
        msg = "instance is not fitted yet, please call fit"
        with pytest.raises(NotFitted, match=msg):
            raise NotFitted.from_estimator(self, 'transform')