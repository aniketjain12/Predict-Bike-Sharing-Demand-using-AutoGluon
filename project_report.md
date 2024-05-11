# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Aniket Jain

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
I realized I need to check my submission.csv file or predictions that the values are not negative so that the kaggle would not reject my submissions. The changes are made as such that all the negative values are replaced with zero but in my model there were no negative values. The other changes was made was when the predictions was saved to the csv files the values were converted into dataframes and the datetime column wants to be added because without it the kaggle was not accepting the submissions and throwing error.

### What was the top ranked model that performed
New datasets:
'leaderboard':                      model   score_val              eval_metric  \
 0      WeightedEnsemble_L3  -53.140047  root_mean_squared_error   
 1   RandomForestMSE_BAG_L2  -53.415263  root_mean_squared_error   
 2          LightGBM_BAG_L2  -55.113793  root_mean_squared_error   
 3        LightGBMXT_BAG_L2  -60.608528  root_mean_squared_error   
 4    KNeighborsDist_BAG_L1  -84.125061  root_mean_squared_error   
 5      WeightedEnsemble_L2  -84.125061  root_mean_squared_error   
 6    KNeighborsUnif_BAG_L1 -101.546199  root_mean_squared_error   
 7   RandomForestMSE_BAG_L1 -116.548359  root_mean_squared_error   
 8          LightGBM_BAG_L1 -131.054162  root_mean_squared_error   
 9          CatBoost_BAG_L1 -131.281712  root_mean_squared_error   
 10       LightGBMXT_BAG_L1 -131.460909  root_mean_squared_error   

 Updated features:
  'leaderboard':                      model   score_val              eval_metric  \
 0      WeightedEnsemble_L3  -54.772618  root_mean_squared_error   
 1          LightGBM_BAG_L2  -54.813447  root_mean_squared_error   
 2        LightGBMXT_BAG_L2  -60.697272  root_mean_squared_error   
 3    KNeighborsDist_BAG_L1  -84.125061  root_mean_squared_error   
 4      WeightedEnsemble_L2  -84.125061  root_mean_squared_error   
 5    KNeighborsUnif_BAG_L1 -101.546199  root_mean_squared_error   
 6   RandomForestMSE_BAG_L1 -116.676952  root_mean_squared_error   
 7     ExtraTreesMSE_BAG_L1 -124.649753  root_mean_squared_error   
 8          LightGBM_BAG_L1 -130.672051  root_mean_squared_error   
 9        LightGBMXT_BAG_L1 -131.174614  root_mean_squared_error   
 10         CatBoost_BAG_L1 -133.269419  root_mean_squared_error   

 Hyperparameter updation:
 'leaderboard':                   model   score_val              eval_metric  pred_time_val  \
 0   WeightedEnsemble_L3 -117.215079  root_mean_squared_error       0.014545   
 1    CatBoost_BAG_L2/T2 -117.615589  root_mean_squared_error       0.012575   
 2    LightGBM_BAG_L2/T1 -117.639856  root_mean_squared_error       0.010014   
 3    CatBoost_BAG_L2/T1 -117.839979  root_mean_squared_error       0.009996   
 4   WeightedEnsemble_L2 -118.056318  root_mean_squared_error       0.010205   
 5   RandomForest_BAG_L1 -118.062730  root_mean_squared_error       0.000242   
 6    LightGBM_BAG_L2/T2 -118.981642  root_mean_squared_error       0.009976   
 7   RandomForest_BAG_L2 -121.630252  root_mean_squared_error       0.010088   
 8    LightGBM_BAG_L1/T2 -131.662831  root_mean_squared_error       0.000184   
 9    LightGBM_BAG_L1/T1 -135.535911  root_mean_squared_error       0.000201   
 10   CatBoost_BAG_L1/T2 -139.265454  root_mean_squared_error       0.008755   
 11   CatBoost_BAG_L1/T1 -140.579611  root_mean_squared_error       0.000185   
 12   CatBoost_BAG_L1/T3 -141.850500  root_mean_squared_error       0.000238   

Here are all models in the first rowsof leaderboards i.e. WeightedEnsemble_L3 is consisting highest ranked model performance. This leaderboard is defined by using,fit_summary() or leaderboard(). In New dataset the  WeightedEnsemble_L3 model scored: -53.140047, In the updated featured dataset the  WeightedEnsemble_L3 model scored: -54.772618 and in Updated hyperparmeter dataset training the  WeightedEnsemble_L3 model scored: -117.215079. These are best model performances respectively for there trainings.


## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
Exploratory data analysis found the demand of bikes on several features(like: weather, season, holiday, etc.) with help of histogram. I added additional features with help of 'dt.month' which defined the feature that in which month the maximum demand of bike is required. 

### How much better did your model preform after adding additional features and why do you think that is?
It didn't performed extraordinarily maybe because of adding one or two new features there is no as such difference in predictions, it needs a large number of varity of the features. The percentage increase to compare the two models: percent increase = final - initial / |initial| * 100
i.e. for model performance the final score was -54.772618 and initial score was -53.140047. So, percent increase = -3.07%
Since the Kaggle score was same for both the models so there is zero percent increase in kaggle score.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
The models predictions values changed very slightly maybe the model needs to be tested on several different hyperparameters for getting some good noticeable performance.The percentage increase to compare the two models: percent increase = final - initial / |initial| * 100
i.e. for model performance the final score was -117.215079 and initial score was -54.772618. So, percent increase = -53.27%
Since the Kaggle score was same for both the models so there is zero percent increase in kaggle score.


### If you were given more time with this dataset, where do you think you would spend more time?
I will be adding more features and testing multiple hyperparameters to improve the performance of the model.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|'time_limit'|'persets'|'eval_metric'|1.80229|
|add_features|'time_limit'|'persets'|'eval_metric'|1.80229|
|hpo|'GBM'|'RF'|'CAT'|1.80229|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_train_score_kaggle.png](img/model_train_score_kaggle.png)

## Summary
The project Predict Bike Sharing Demand was very interesting, we learnt how to use kaggle and kaggle datasets and make appropriate prediction models on it, with help of AutoGluon which is very easy and useful framework. First of all, we loaded the required frameworks like mxnet and autogluon, then downloaded the dataset from the kaggle after that we loaded the dataset and described it, after that we trained the model 3 times first time with datasets with no changes, second time added feature and third time added new hyperparameters. After that we predicted values with the test dataset respectively individually for three times. After that we added the prediction value in submission.csv files and submitted to the kaggle competition. Then we done some Exploratory data analysis (EDA) steps and plot some graphs for features and scores of each model for comparison and then made the report.  


