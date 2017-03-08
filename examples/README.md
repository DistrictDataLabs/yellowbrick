# Yellowbrick Examples 

[![Visualizers](../docs/images/visualizers.png)](../docs/images/visualizers.png)

Welcome to the yellowbrick examples directory! This directory contains a gallery of visualizers and their application to classification, regression, clustering, and other machine learning techniques with Scikit-Learn. Examples have been submitted both by the Yellowbrick team and also users like you! The result is a rich gallery of tools and techniques to equip your machine learning with visual diagnostics and visualizer workflows! 

## Getting Started 

The notebook to explore first is the `examples.ipynb` Jupyter notebook. This notebook contains the executable examples from the tutorial in the documentation. However, before you can successfully run this notebook, you must first download the sample datasets. To download the samples run the downloader script:

```
$ python download.py 
```

This should create a directory called `examples/data`, which in turn will contain CSV or text datasets. There are two primary problems that the download script may have: first, you may get the error `"The requests module is required to download data"`. To fix this problem:

```
$ pip install requests 
```

The second problem may be `"Download signature does not match hardcoded signature!"` This problem means that the file you're trying to download has changed. Either download a more recent version of Yellowbrick, or use the URLs in the `download.py` script to fetch the data manually. If there are any other problems, please notify us via [GitHub Issues](https://github.com/DistrictDataLabs/yellowbrick/issues). 

Once the example data has been downloaded, you can run the examples notebook as follows:

```
$ jupyter notebook examples.ipynb 
```

If you don't have jupyter installed, or other dependencies, you may have to `pip install` them. 

## Organization 

The examples directory contains many notebooks, folders and files. At the top level you will see the following:

- examples.ipynb: a notebook with executable versions of the tutorial visualizers 
- download.py: a script to download the example data sets 
- palettes.ipynb: a visualization of the Yellowbrick palettes 
- data: a directory containing the example datasets. 

In addition to these files and directory, you will see many other directories, whose names are the GitHub usernames of their contributors. You can explore these user submitted examples or submit your own! 

### Contributing 

To contribute an example notebook of your own, perform the following steps:

1. Fork the repository into your own account 
2. Checkout the develop branch (see [contributing to Yellowbrick](http://www.scikit-yb.org/en/latest/about.html#contributing) for more. 
3. Create a directory in the repo, `examples/username` where username is your GitHub username. 
4. Create a notebook in that directory with your example. See [user testing](http://www.scikit-yb.org/en/latest/evaluation.html) for more. 
5. Commit your changes back to your fork. 
6. Submit a pull-request from your develop branch to the Yellowbrick develop branch. 
7. Complete the code review steps with a Yellowbrick team member. 

That's it -- thank you for contributing your example! 

A couple of notes. First, please make sure that the Jupyter notebook you submit is "run" -- that is it has the output saved to the notebook and is viewable on GitHub (empty notebooks don't serve well as a gallery). Second, please do not commit datasets, but instead provide instructions for downloading the dataset. You can create a downloader utility similar to ours. 

One great tip, is to create your PR right after you fork the repo; that way we can work with you on the changes you're making and communicate about how to have a very successful contribution! 

### User Examples 

In this section we want to thank our examples contributors, and describe their notebooks so that you can find an example similar to your application! 

- [bbengfort](https://github.com/bbengfort): visualizing text classification 
- [rebeccabilbro](https://github.com/rebeccabilbro): visualizing book reviews data 
- [nathan](https://github.com/ndanielsen/): visualizing the Iris dataset 
