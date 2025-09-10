best_nmse_cv = search.best_score_
print(f"Best score during the RandomizedSearchCV: {best_nmse_cv:.4f}")
if self.best_acc_ is None: 
    self.best_acc_ = best_nmse_cv
    self.model = search.best_estimator_
    self.best_params_ = search.best_params_ 
else:
    if best_nmse_cv < 1.1*self.best_acc_:
        self.best_acc_ = best_nmse_cv
        self.model = search.best_estimator_         # recover the best model
        self.best_params_ = search.best_params_     # recover the best hp
        print("🆕 Model has been changed.")

# cv_results = search.cv_results_
# for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
#     print(f"R²: {mean_score:.4f} -> {params}")