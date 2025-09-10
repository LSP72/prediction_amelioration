from SVRModel import SVRModel

SVR_VIT = SVRModel()
SVR_END = SVRModel()
for i in range(len(X_train)):
    # Adding training data in the speed model
    SVR_VIT.add_data(X_train.iloc[i].to_frame().T, [Y_train_VIT.iloc[i]])
    # Adding training data in the endurance model    
    SVR_END.add_data(X_train.iloc[i].to_frame().T, [Y_train_END.iloc[i]])
    
# Testing with new data 
from sklearn.metrics import r2_score, mean_squared_error

# Predictions on training set
y_train_pred = SVR_VIT.model.predict(SVR_VIT.X)
r2_train = r2_score(SVR_VIT.y, y_train_pred)
mse_train = mean_squared_error(SVR_VIT.y, y_train_pred)

# Predictions on test set
y_test_pred = SVR_VIT.model.predict(X_test)  # make sure X_test is defined
r2_test = r2_score(Y_test_VIT, y_test_pred)
mse_test = mean_squared_error(Y_test_VIT, y_test_pred)

# Print results nicely
print("📊 Training Metrics:")
print(f"R²: {r2_train:.4f}, MSE: {mse_train:.4f}\n")

print("📊 Test Metrics:")
print(f"R²: {r2_test:.4f}, MSE: {mse_test:.4f}")
print(f"🔸 Predicted values:{y_test_pred}")
true_values = Y_test_VIT['VIT_POST'].to_list()
print(f"🔹 True values: {true_values}")

#%% Testing with different testing dataset
from SVRModel import SVRModel
import random as rdm
import pandas as pd

VIT_POST = all_data['VIT_POST']
END_POST=all_data['6MWT_POST']
data = all_data.drop(columns=["VIT_POST", "6MWT_POST"])

results_VIT, results_END = [], []

n = len(data)

for j in range(1):
    # Choosing which row will be used for testing
    a, b = rdm.randint(0, n-1), rdm.randint(0, n-1)
    if a == b and b != n-1:
        b = a + 1
    elif a == b and b != n-1:
        b = a -1
    else:
        pass
    
    # Creating the feature data sets (for training and testing)
    X_test = data.iloc[[a, b]]
    X_train = data.drop(data.index[[a, b]])
    
    # Creating the label data sets (training and testing)
    Y_test_VIT = VIT_POST.iloc[[a, b]]
    Y_test_END = END_POST.iloc[[a, b]]
    Y_train_END = END_POST.drop(END_POST.index[[a, b]])
    Y_train_VIT = VIT_POST.drop(VIT_POST.index[[a, b]])
    
    # Creating the models 
    SVR_VIT = SVRModel()
    SVR_END = SVRModel()
    
    for i in range(len(X_train)):
        # Adding training data in the speed model
        SVR_VIT.add_data(X_train.iloc[i].to_frame().T, [Y_train_VIT.iloc[i]])
        # Adding training data in the endurance model    
        SVR_END.add_data(X_train.iloc[i].to_frame().T, [Y_train_END.iloc[i]])
    
    # Training the models
    SVR_VIT.train_and_tune(100)
    SVR_END.train_and_tune(100)
    
    # Predicitons
    y_test_pred_VIT = SVR_VIT.model.predict(X_test)
    y_test_pred_END = SVR_END.model.predict(X_test)
    
    mse_test_VIT = mean_squared_error(Y_test_VIT, y_test_pred_VIT)
    mse_test_END = mean_squared_error(Y_test_END, y_test_pred_END)
    
    # Storing the results
    for i in range(len(y_test_pred_VIT)):
        results_VIT.append({
                'True': Y_test_VIT.iloc[i],
                'Prediction': y_test_pred_VIT[i],
                'Delta': Y_test_VIT.iloc[i] - y_test_pred_VIT[i],
                'MSE': mse_test_VIT
            })
        
    for i in range(len(y_test_pred_END)):
        results_END.append({
                'True': Y_test_END.iloc[i],
                'Prediction': y_test_pred_END[i],
                'Delta': Y_test_END.iloc[i] - y_test_pred_END[i],
                'MSE': mse_test_END
            })
    
    results_VIT_df = pd.DataFrame(results_VIT)
    results_END_df = pd.DataFrame(results_END)
