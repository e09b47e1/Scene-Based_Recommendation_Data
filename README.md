# Datasets of Scene-Based Recommendation Task (for reviewers) #

These are anonymous open source datasets for scene-based recommendation task.
You can find a more concrete description about the datasets in Section Experiment of our submitted paper.
In order to comply with the submission policy of the conference, there is no description of content except file format here.
The datasets will be released formally in camera ready submission. 

## Datasets Introduction  ##
We construct two different datasets based on two major category commodity sets with different features:

- Baby&Toy: commodities in this dataset are mainly for the pregnant women, new mothers, toddlers and infants.
- Electronics: these commodities include digital products, computer peripherals, etc.



## Files in Each Dataset ##
The file structure in each dataset is the same. We will introduce these files one by one:

- user\_item\_pair.txt: the interactions between users and commodities. There are three elements in each row: a user, a commodity, and a boolean.
- scene-cate\_set.txt: the scenes which consist of categories. Each row is a scene and there are several categories in it.
- cate\_cate\_pair.txt: the relevance between categories. There are three elements in each row: category 1, category 2, and their relevance score.
- item\_cate\_pair.txt: the relationship between commodities and categories. There are three elements in each row: a commodity, a category, and boolean which indicates whether the commodity belongs to the category.
- item\_item\_pair.txt: the similarity between commodities. There are three elements in each row: commodity 1, commodity 2, and normalized similarity scores.







