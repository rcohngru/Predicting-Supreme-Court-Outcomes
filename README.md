# Predicting Supreme Court Outcomes
Using machine learning to predict the outcomes of Supreme Court cases.

## Why the Supreme Court?
The Supreme Court is the arbiter of law in the United States. Beginning with the opinion of West v. Barnes in 1791, the Court has defended the Constitution and American people for over 200 years. Decisions made in this Court have long-lasting implications that affect hundreds of millions of people. As someone who follows politics and the Court I decided that attempting to predict outcomes of cases would be a good exercise in using NLP while still finding the topic interesting.

## Data Acquisition
__Transcripts__:
    I sourced the transcript data from EricWeiner's [github repository](https://github.com/EricWiener/supreme-court-cases). The data was originally provided by Oyez, an "unofficial online multimedia archive of the Supreme Court of the United States". Each file corresponds to one case formatted in JSON, containing relevant case information (docket, term, decision data, etc...) and the transcript of the oral argument.
    
__Decisions__:
    The Oyez data does not come with the actual decisions each justice makes per case. However, the Washington University School of Law maintains a [database](http://scdb.wustl.edu/data.php) that provides justice-oriented decisions--each row in the table corresponds to one justice's decision per case, including information such as the winning party, whether the justice voted with the majority, and the political direction the justice leaned.
    
## Data Formatting & Cleaning

<p align="center">
  <img width="900" height="400" src="img/data_reformatting.png">
</p>

The Oyez data I started with was JSON formatted which is not particularly useful in this case. For each case I extracted relevant case information (docket, term, decision date, etc...) and the transcript of the oral arguments, and inserted it into a CSV file. The transcripts are broken up into portions depending on who is speaking. To start, I decided to make each portion its own row in the CSV. This is the [transcripts_by_justice.csv](https://github.com/rcohngru/Predicting-Supreme-Court-Outcomes/blob/master/data/transcripts_by_justice.csv) file.

I then created CSV files for each justice where each row corresponds to a unique case. I concatenated all of the separate speaking moments for each case, so each row contains a document representing every word the justice spoke during the oral arguments. Conveniently, the Oyez dataset also contains pre-cleaned versions of the transcripts meaning that I do not have to bother with it.

Finally, I joined the justice CSVs with the decisions data from Washington University. Interestingly, the Washington University did not contain explicit data for how each justice voted--rather the data it contained was if the justice voted with the majority, and if the petitioning party won the case. I converted this data into a vote column using the following schema.

|                | 0                                       | 1                                   |
|----------------|-----------------------------------------|-------------------------------------|
| `vote`         | voted against petitioner                    | voted for petitioner            |
| `partyWinning` | no favorable disposition for petitioner | favorable dispostion for petitioner |
| `majority`     | justice dissented                       | justice voted with majority         |


|                 | `partyWinning` == 0 | `partyWinning` == 1 |
|-----------------|---------------------|---------------------|
| `majority` == 0 | `vote` = 1          | `vote` = 0          |
| `majority` == 1 | `vote` = 0          | `vote` = 1          |

The `vote` column is what I will be trying to predict.

## Initial EDA

With all of the data cleaned and formatted in the way that I wanted it to be, I began conducting some basic exploration with it to get a better feel for what it looks like.

<p align="center">
  <img width="900" height="450" src="img/cases_per_justice.png">
</p>

The longest serving justices have heard the most cases, as is to be expected. Looking at this graph it is apparent that correctly predicting the votes for Justices Gorsuch and Kavanaugh may be be more challening due to lack of data. As the two newest justices--both appointed within the last few years--they have had fewer opportunities to speak.

<p align="center">
  <img width="900" height="450" src="img/words_spoken.png">
</p>

The plot of the total number of words spoken by each Justice is similar to the number of cases, with one notable difference: Clarence Thomas. Thomas is commonly known as the 'Silent Justice' because of how infrequently he speaks during oral arguments. Fun fact: Justice Thomas once went 10 years without asking a single question during a session of the Supreme Court. There is also a smaller amount of data for Justices Gorsuch and Kavanaugh--this too can be attributed to how recently they were nominated to the position.

<p align="center">
  <img width="900" height="900" src="img/vote_distribution.png">
</p>

The way each Justice votes is remarkably similar, despite the differences in their political idealogies. Each Justice tends to vote in favor of the petitioner, although this does not mean they all vote this way at the same time. While I am not sure why this is the case, this does help establish a baseline for my machine learning models. The baseline would be to always predict in favor of the petitioner. Because the classes are imbalanced, the evaluation metrics most relevant are precision and recall. For the baseline, the recall value would be 1.0 and the precision value would be ~0.6.

<p align="center">
  <img width="800" height="560" src="img/court_decisions.png">
</p>

Using data from when Brett Kavanaugh joined the Supreme Court and onward we see that the court as a whole follows the same trend of favoring the petitioner. This also helps in establishing a baseline for the ensemble model to predict the outcome of each case. If I were to always predict that the Court votes in favor of the petitioner, I would get recall and precision values 0f 1.0 and 0.6, respectively.

To avoid redundancy in this README, going forward I will only use Justice Breyer's data to illustrate my process. Having spoken the most words out of any of the other Justices, his data will be the most useful.

## Data Adjustments

The previous graphs tell me that there is a significant imbalance in the class distribution of the data. This is a problem because any model that I use will tend to skew towards the majority class simply because there is more of it. I don't want the *amount* of data to influence a classification, I want the content of the data to be the determining factor in a classification.

<p align="center">
  <img width="900" height="350" src="img/over_under_viz.png">
</p>

One method of accounting for a class imbalance in the data is to use **undersampling**. The concept behind undersampling is to randomly exclude some data in the majority class from the training set, so that the class distribution of the data follows an even 50/50 split. The key problem with undnersampling is that you end up with less data overall, which is never ideal.

Another method of accounting for imbalance is to use **oversampling**. Oversampling works in a very similar manner to undersampling, the key difference being that in this case data from the minority class is randomly duplicated to match the number of observations in the majority class. While this helps prevent scarcity of data being a problem as in undersampling, another issue presents itself: the presence of duplicate data in the minority class may improperly influence the model.

**SMOTE** (Synthetic Minority Oversampling Technique), is a method of balancing that seeks to fix the problem with oversampling. Rather than randomly duplicating data in the minority class, SMOTE algorithmically creates synthetic data from the minority class, so that the presence of duplicates does not unduly weight the model.

In the original SMOTE paper, the authors recommended trying a combination of SMOTE and undersampling, so I will try that as well.

<p align="center">
  <img width="900" height="350" src="img/breyer_nobalance.png">
</p>

<p align="center">
  <img width="900" height="350" src="img/breyer_oversample.png">
</p>

<p align="center">
  <img width="900" height="350" src="img/breyer_undersample.png">
</p>

<p align="center">
  <img width="900" height="350" src="img/breyer_smote.png">
</p>

<p align="center">
  <img width="900" height="350" src="img/breyer_both.png">
</p>

Clearly, either no balancing or SMOTE seem to be the best balancing methods to use in this situation. The precision and recall values of the data are consistently higher in these two cases than when using the others. Going forward, I will use both SMOTE and no rebalancing when training my models.

## Modeling

When determing which balancing method to use, I evaluted the results using the `sklearn` defaults of four classifications models: Logistic Regression, Random Forests, Gradient Boosting, and Support Vecotr Machine Classifiers. Examining the plot for undersampling, it is not immediately obvious which, if any, model is superior. 

Because of this, I decided to tune each of the models to see how precise I could get them to be.

<p align="center">
  <img width="900" height="350" src="img/breyer_lr.png">
</p>

<p align="center">
  <img width="900" height="350" src="img/breyer_rf.png">
</p>

<p align="center">
  <img width="900" height="350" src="img/breyer_gb.png">
</p>

<p align="center">
  <img width="900" height="350" src="img/breyer_svc.png">
</p>

After running a `GridSearchCV` model to find the optimal set of hyperparameters for all four types of Classifiers with both no rebalancing and SMOTE, it's clear that there is not much of an improvement between the default `sklearn` model and the optimized one, with the exceptions being the SVM and Logistic Regression Classifier in terms of Recall.

| Model w/ No Balance | Precision | Recall  | Model w/ SMOTE | Precision | Recall  |
|---------------------|-----------|---------|----------------|-----------|---------|
| Logistic Reg.       | 0.61479   | 0.99371 | Logistic Reg.  | 0.62360   | 0.68944 |
| Default RF          | 0.61508   | 0.97484 | Default RF     | 0.62661   | 0.90683 |
| Optimized RF        | 0.61508   | 0.97484 | Optimized RF   | 0.62447   | 0.91925 |
| Default GB          | 0.60811   | 0.84906 | Default GB     | 0.62564   | 0.75776 |
| Optimized GB        | 0.61628   | 1.00000 | Optimized GB   | 0.62753   | 0.96273 |
| Default SVC         | 0.61628   | 1.00000 | Default SVC    | 0.61572   | 0.87578 | 
| Optimized SVC       | 0.61628   | 1.00000 | Optimized SVC  | 0.62601   | 0.95652 |

## Results

At this point, I decided to move ahead using a SVC for my final testing. I opted to test with both SMOTE and no balancing, the results for No Balancing are shown below:

<p align="center">
  <img width="900" height="900" src="img/justice_matrix_test.png">
</p>

As can be seen, it turns out the models for each justice did exactly what I was hoping they would not--predict exclusively the majority class for the output. Unfortunately, this is the case for every model I tried, including densely connected neural networks. 

<p align="center">
  <img width="450" height="450" src="img/court_matrix_test.png">
</p>

I decided to ensemble the justice predictions anyways to predict the outcome of the cases and got a similar result. 

## Next Steps

It seems that I have reached the limit of the effectiveness of my data. I tried a number of different types of models, all with similar results. Going forward, I would want to supplement my data with the easily quantifiable following information:
    - Ideology of each justice
    - Number of times spoken
    - Duration of speaking time
    - Speaking to petitioner or respondent
    
Furthermore, I excluded the transcript data of everyone who was not a Justice during the oral arguments. I would like to find a way to incorporate this information both at the Justice-level and case-level for future models.

Finally, I think it would be wise to pull data from the lower courts for Kavanaugh and Gorsuch. Being relatively new Justices, they have not spoken much, but I think supplementing the data with this information may be useful. Unfortunately, the same case can not be made for Clarence Thomas. If I ever meet him, I will advise him to ask more questions so that I can supplement my data with it.