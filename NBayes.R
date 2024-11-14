library(tidyverse)
library(tidymodels)
library(vroom)
library(themis)
library(embed)
library(doParallel)
library(discrim)

num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

train <- vroom('train.csv')
train$color <- factor(train$color)
train$type <- factor(train$type)
test <- vroom('test.csv')

ghouls_recipe <- recipe(type ~ ., data = train) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_range(all_numeric_predictors(),min = 0, max = 1) %>%
  step_smote(all_outcomes(), neighbors = 5)
prep <- prep(ghouls_recipe)
bake <- bake(prep, new_data = train)

my_mod <- naive_Bayes(Laplace=tune(), 
                      smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(ghouls_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 20) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 10, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc,accuracy)) #, f_meas, sens, recall, spec, precision, accuracy)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
ghoul_predictions <- final_wf %>%
  predict(new_data = test, type="class") # "class"(yes or no) or "prob"(probability)

kaggle_submission <- ghoul_predictions %>%
  bind_cols(., test) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)

## Write out the file
vroom_write(x=kaggle_submission, file="./nbpreds.csv", delim=",")

stopCluster(cl)

