library(tidyverse)
library(tidymodels)
library(vroom)
library(themis)
library(embed)
library(doParallel)

num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

train <- vroom('train.csv')
train$color <- factor(train$color)
train$type <- factor(train$type)
test <- vroom('test.csv')

ghouls_recipe <- recipe(type ~ ., data = train) |>
  step_impute_knn(bone_length, impute_with = imp_vars(has_soul, color), neighbors = 7) |>
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul, color, bone_length), neighbors = 8) |>
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul, color, bone_length, rotting_flesh), neighbors = 6)
prep <- prep(ghouls_recipe)
bake <- bake(prep, new_data = train)

my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(ghouls_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,100)),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc,accuracy)) #, f_meas, sens, recall, spec, precision, accuracy)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- forest_wf %>%
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
vroom_write(x=kaggle_submission, file="./forestpreds.csv", delim=",")

stopCluster(cl)




