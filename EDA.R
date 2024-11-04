library(tidyverse)
library(tidymodels)
library(vroom)
library(themis)

train <- vroom('train.csv')
train$color <- factor(train$color)
trainVals$type <- factor(train$type)
test <- vroom('test.csv')
missingVals <- vroom('trainWithMissingValues.csv')
missingVals$color <- factor(missingVals$color)
missingVals$type <- factor(missingVals$type)

ghouls_recipe <- recipe(type ~ ., data = missingVals) |>
  step_impute_knn(bone_length, impute_with = imp_vars(has_soul, color), neighbors = 7) |>
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul, color, bone_length), neighbors = 8) |>
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul, color, bone_length, rotting_flesh), neighbors = 6)
prep <- prep(ghouls_recipe)
bake <- bake(prep, new_data = missingVals)

rmse_vec(train[is.na(missingVals)],
         bake[is.na(missingVals)])

