# Yellowbrick Tests

*Welcome to the Yellowbrick tests!*

If you're looking for information about how to use Yellowbrick, for our contributor's guide, for examples and teaching resources, for answers to frequently asked questions, and more, please visit the latest version of our documentation at [www.scikit-yb.org](https://www.scikit-yb.org/).

## Running Yellowbrick Tests

To run the tests locally, first install the tests-specific requirements with `pip` using the `requirements.txt` file in the `tests` directory:

```
$ pip install -r tests/requirements.txt
```

The required dependencies for the test suite include testing utilities and libraries such as `pandas` and `nltk` that are not included in the core dependencies.

Tests can then be run as follows from the project `root`:

```bash
$ make test
```

The Makefile uses the `pytest` runner and testing suite as well as the coverage library.

## Adding a Test for Your Visualizer

The `tests` package mirrors the yellowbrick package in structure and also contains several helper methods and base functionality. To add a test to your visualizer, find the corresponding file to add the test case, or create a new test file in the same place you added your code.

### Visual Tests

The primary test you should create is simply to test your visualizer from end to end and make sure that no exceptions occur.

Visual tests are notoriously difficult to create --- how do you test a visualization or figure? Moreover, testing scikit-learn models with real data can consume a lot of memory. To assist with this, we have two primary helpers, `VisualTestCase` and the `yellowbrick.datasets` module.

Leverage these helpers to create your tests as follows:

```python
import pytest

from tests.base import VisualTestCase
from yellowbrick.datasets import load_occupancy


class MyVisualizerTests(VisualTestCase):

    def test_my_visualizer(self):
        """
        Test MyVisualizer on a real dataset
        """
        # Load the data using the Yellowbrick datasets module
        X, y = load_occupancy()

        try:
            visualizer = MyVisualizer()
            visualizer.fit(X)
            visualizer.finalize()
        except Exception as e:
            pytest.fail("my visualizer didn't work")
```

### Image Comparison Tests

Writing an image-based comparison test is only a little more difficult than the simple test case presented above. We have adapted `matplotlib`'s image comparison test utility into an easy to use assert method: `self.assert_images_similar(visualizer)`

The main consideration is that you must specify the “baseline” (i.e. expected) image in the `tests/baseline_images/` folder structure.

For example, let's say you create your tests in `tests/test_regressor/test_myvisualizer.py` as follows:

```python
from tests.base import VisualTestCase
...
    def test_my_visualizer_output(self):
        ...
        visualizer = MyVisualizer()
        visualizer.fit(X)
        visualizer.finalize()
        self.assert_images_similar(visualizer)
```

The first time this test is run, there will be no baseline image to compare against, so the test will fail. Alternatively, if you are making a correction to the existing test `test_my_visualizer_output`, and the correction modifies the resulting test image, the test may also fail to match the existing baseline image. The solution is to first run the tests, then copy the new output images to the correct subdirectory under source code revision control (with `git add`). When rerunning the tests, they should now pass!

We have a helper script, `tests/images.py` to clean up and manage baseline images automatically. It is run using the ``python -m`` command to execute a module as main, and it takes as an argument the path to **your** test file. To copy the figures as above:

```bash
$ python -m tests.images tests/test_regressor/test_myvisualizer.py
```

This will move all related test images from `actual_images` to `baseline_images` on your behalf (note you'll have had to already run the tests at least once to generate the images). You can also clean up images from both actual and baseline as follows:

```bash
$ python -m tests.images -C tests/test_regressor/test_myvisualizer.py
```

This is useful particularly if you're stuck trying to get an image comparison to work. For more information on the images helper script, use `python -m tests.images --help`.
