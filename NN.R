library(tidyverse)
library(tidymodels)
library(vroom)
library(themis)
library(embed)

train <- vroom('train.csv')
train$color <- factor(train$color)
train$type <- factor(train$type)
test <- vroom('test.csv')

ghouls_recipe <- recipe(type ~ ., data = train) %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1)
prep <- prep(ghouls_recipe)
bake <- bake(prep, new_data = train)

my_mod <- mlp(hidden_units = tune(),
              epochs = 50) %>%
  set_engine("keras") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(ghouls_recipe) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(hidden_units(range=c(1, 50)),
                            levels=5)

folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc,accuracy))

CV_results %>% collect_metrics() %>%
  filter(.metric=="roc_auc") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- nn_wf %>%
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
vroom_write(x=kaggle_submission, file="./nnpreds.csv", delim=",")

