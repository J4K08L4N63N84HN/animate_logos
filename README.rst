SVG Logo Animation using Machine Learning
-----------------------------------------

.. image:: https://img.shields.io/github/repo-size/J4K08L4N63N84HN/animate_logos   :alt: GitHub repo size
.. image:: https://img.shields.io/github/license/J4K08L4N63N84HN/animate_logos   :alt: GitHub
.. image:: https://img.shields.io/github/stars/J4K08L4N63N84HN/animate_logos?style=social   :alt: GitHub Repo stars


This project allows to automatically animate logos in SVG format using machine learning.

Its functionality includes:

* Extract SVG information, e.g., size, position, color
* Get SVG embeddings of logos by using deepSVG's hierarchical generative network
* Automatically animate logos using two different approaches: Genetic algorithm and entmoot optimizer

The project started in November 2020 as a Masters Team Project at the University of Mannheim.

Table of Contents
#################

.. contents::

Description
#################


* What your application does,
* Why you used the technologies you used,
* Some of the challenges you faced and features you hope to implement in the future.


How to Install
##############
If your project is software or an app that needs installation, you should include the steps required to install your project. Provide a step-by-step description of how to get the development environment running.

.. code:: python

    from src.pipeline import Logo
    logo = Logo(data_dir='path/to/my/svgs/logo.svg')
    logo.animate()

Detailed documentation and usage instructions can be found `here <https://animate-logos.readthedocs.io/en/latest/>`__.


How to Use
##########

.. code:: python

    from src.pipeline import Logo
    logo = Logo(data_dir='path/to/my/svgs/logo.svg')
    logo.animate()

Detailed documentation and usage instructions can be found `here <https://animate-logos.readthedocs.io/en/latest/>`__.




Provide instructions and examples so users/contributors can use the project. This will make it easy for them in case they encounter a problem – they will always have a place of reference.
You can also include screenshots to show examples of the running project.

The data is collected through a `labeling website <https://animate-logos.web.app/>`__ (`Github <https://github.com/J4K08L4N63N84HN/animate_logos_label_website>`__) where users can rate the quality of animations.



Credits
#######

If you worked on the project as a team or an organization, list your collaborators/team members. You should also include links to their GitHub profiles.
Also, if you followed tutorials to build that particular project, include links to those here as well. This is just a way to show your appreciation and also to help others get a first hand copy of the project.

License
#######

This is the last section of most README. It lets other developers know what they can and cannot do with your project. If you need help choosing a license, use https://choosealicense.com/
🏆 The sections listed above are the minimum for a good README. But you might also want to consider adding the following sections.







