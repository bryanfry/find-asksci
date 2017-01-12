# README


**[find-asksci](http://34.196.238.24)** is a web application that aims to locate the redditor in the subreddit /r/askscience who is best-qualified to answer a user's technical question.  I developed this application as my final project at the Metis Data Science Bootcamp in NYC.  This project involved web scraping, natural language processing (NLP), dimensionality reduction, machine learning, visualization, and Flask app development.

As of 1/11/17, the app is hosted on an Amazon Web Services EC2 instance.  Please click [here](http://34.196.238.24) to give it a try!   

_(**HINT:** asking more detailed questions including technical / scientific terminology will usually give the best results.)_

This repository includes the development and production code for the find-asksci app. **Five iPython notebooks** (numbered 01 - 05) illustrate the scraping, data manipulation, and machine learning yielding the models behind the web application.  The first two notebooks were run locally on my MacBook.  I ran notebooks 03 - 05 on a compute-optimized EC2 instance with 30 GB RAM, as they were too memory-intensive to execute successfully  on my local machine.  The web application itself is a Python Flask app (hosted with the Apache webserver) that relies on large pickled data structures produced by the initial iPython notebooks.  The HTML for the webpage is also included.  


## Dependencies (development and production code)

* pandas
* numpy
* Selenium
* MongoDB + pymongo
* sklearn
* matplotlib + seaborn
* Flask  
* Online access to Scopus database of academic journals

## More info?
For discussion of the design and implementation, please see the document **discussion-find-asksci.md** in this repository.