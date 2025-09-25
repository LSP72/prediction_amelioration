# prediction_amelioration
This Python package has been created to try to use machine learning (ML) for predicting amélioration in gait in children with CP. The latter could also be used for other prediction purposes based on similar variables.

Functioning:
In the folder “examples”, can be found different running srcipt with different purposes. However, for each, here is the step by step process:
- first, it will load the data (tho ses ones should be given as a matrix with features and labels in columns and samples in rows)
- second, it will extract the features and the labels
    - For classification predictions, it will also calculate the labels from the pre/post difference of the evaluated outcome
- third, it will initiate, optimise and train a model on a training set (80%)
- fourth, it will give the prediction of the testing set (20%) and their metrics (accuracy, f1 score, recall, …)
- fifth, it will plot the shap analysis to understand the weight of each feature in the ML’s decision making

Several models (whether regressor or classifier) are accessible in the folder “models”.
