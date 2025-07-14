library(xgboost)
library(tidyverse)
#install.packages("pROC")
library(pROC)      # For ROC curves and model evaluation
library(randomForest)
#install.packages("pdp")
library(pdp)
#install.packages("SHAPforxgboost")
library(SHAPforxgboost)



#--------------------------------------------------------------
# Read and prepare the data
#--------------------------------------------------------------

vicData <- read_csv("https://raw.githubusercontent.com/OpenActTextDev/ActuarialRegression/refs/heads/main/CourseCSVData/VicRoadFatalData.csv")

str(vicData) # check the data

#Remove variables 
vicDataPre <-  vicData %>% 
  select(-DRIVER_ID,   
         -VEHICLE_ID,     
         -OWNER_POSTCODE, #It has too many levels
         -ACCIDENT_NO, 
         - DAY_OF_WEEK,
         -fatal, 
         -accident_cnt)


#Create new variables based on data and time

vicDataPre <- vicDataPre %>% 
  mutate(hour = (hour(ACCIDENTTIME)), 
         ACCIDENTDATE = as.Date(ACCIDENTDATE, format = "%d/%m/%Y"),
         month = (month(ACCIDENTDATE)),
         year = (year(ACCIDENTDATE))) %>% 
  select(-ACCIDENTTIME, -ACCIDENTDATE, -`Age Group`)

#Convert character variables to factor
vicDataPre[] <- lapply(vicDataPre, function(x) if(is.character(x)) as.factor(x) else x)


#--------------------------------------------------------------
# Split the data into training and testing
#--------------------------------------------------------------

n <- nrow(vicDataPre)
set.seed(123)

indexTrain <- sample(1:n, round(n*0.8)) #Do 80/20 split

vicDataTrain <- vicDataPre[indexTrain, ] 
vicDataTest <- vicDataPre[-indexTrain, ] 




#--------------------------------------------
# Random Forest
#--------------------------------------------

set.seed(10)
RFModel <- randomForest(factor(fatal_cnt) ~ .,
                        data= vicDataTrain, 
                        importance=TRUE, 
                        ntree=50,
                        xtest = vicDataTest %>% 
                          select(-fatal_cnt), 
                        ytest = factor(vicDataTest$fatal_cnt))

#Plot out of bag error rate
plot(RFModel$err.rate[,1], xlab = "Trees", ylab = "OOB error rate")

#Plot variable importance
varImpPlot(RFModel)

#Model eveluation
predTestRF <- RFModel$test$votes[, 2]
ROC_test_RF <- roc(vicDataTest$fatal_cnt, predTestRF)
# Plot ROC curves
plot(ROC_test_RF, col = "blue", main = "ROC Curves for Random Forest", print.auc = TRUE)
#generate AUC
auc(ROC_test_RF)  


#--------------------------------------------------------------
# Prepare data for XGBoost: Do a further split for validation
#--------------------------------------------------------------

# Prepare and clean data
full_data <- vicDataTrain 

# Create a 80/20 training/validation split
set.seed(23)
n_test <- nrow(full_data)
train_idx <- sample(1:n_test, round(n_test*0.8)) #Do 80/20 split
train_set <- full_data[train_idx, ]
valid_set <- full_data[-train_idx, ]

# Create matrix inputs
X_train <- model.matrix(fatal_cnt ~ . - 1, data = train_set)
y_train <- as.numeric(train_set$fatal_cnt)

X_valid <- model.matrix(fatal_cnt ~ . - 1, data = valid_set)
y_valid <- as.numeric(valid_set$fatal_cnt)

# Convert to DMatrix
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dvalid <- xgb.DMatrix(data = X_valid, label = y_valid)

# Set watchlist to monitor both sets
watchlist <- list(train = dtrain, eval = dvalid)

xgb_model <- xgb.train(
  data = dtrain,
  objective = "binary:logistic",
  nrounds = 500,
  eval_metric = "logloss",
  watchlist = watchlist,
  early_stopping_rounds = 10,
  verbose = 1
)

eval_log <- xgb_model$evaluation_log
best_iter <- xgb_model$best_iteration


#Plot training and validation log loss
plot(eval_log$iter, eval_log$train_logloss,
     type = "l", col = "blue", lwd = 2,
     xlab = "Boosting Iteration",
     ylab = "Log Loss",
     ylim = range(c(eval_log$train_logloss, eval_log$eval_logloss)),
     main = "Training vs. Validation Log Loss")

lines(eval_log$iter, eval_log$eval_logloss, col = "red", lwd = 2)
abline(v = best_iter, col = "darkgreen", lty = 2)

legend(x = 5, y=0.44, legend = c("Training", "Validation", paste("Best Iter =", best_iter)),
       col = c("blue", "red", "darkgreen"), lwd = c(2, 2, 1), lty = c(1, 1, 2))

#--------------------------------------------------------------
# Train final model on the full training set
#--------------------------------------------------------------

# Full training set
X_full <- model.matrix(fatal_cnt ~ . - 1, data = full_data)
y_full <- as.numeric(full_data$fatal_cnt)

dtrain_full <- xgb.DMatrix(data = X_full, label = y_full)

final_model <- xgb.train(
  data = dtrain_full,
  objective = "binary:logistic",
  nrounds = best_iter,
  eval_metric = "logloss",
  verbose = 0
)


#--------------------------------------------------------------
# Model predictions and evaluation
#--------------------------------------------------------------

# Prepare test data
test_set <- vicDataTest
# Create matrix inputs
X_test <- model.matrix(fatal_cnt ~ . - 1, data = test_set)
y_test <- as.numeric(test_set$fatal_cnt)

# Convert to DMatrix
dtest <- xgb.DMatrix(data = X_test, label = y_test)

predTestxgboost <- predict(final_model, newdata = dtest)

#ROC
library(pROC)
ROC_test_xgb <- roc(vicDataTest$fatal_cnt, predTestxgboost)
# Plot ROC curves
plot(ROC_test_xgb, col = "green", main = "ROC Curves for XGBoost", print.auc = TRUE)
ROC_test_xgb$auc #Get AUC value
  

#--------------------------------------------------------------
# Feauture Importance
#--------------------------------------------------------------

xgbVarI <- xgb.importance(model = final_model)

xgb.plot.importance(xgbVarI, top_n = 20, rel_to_first = TRUE)


#--------------------------------------------------------------
# Interpretability with Partial Dependence Plots (PDP)
#--------------------------------------------------------------


# Convert matrix to data.frame for interpretability
X_df <- as.data.frame(X_full)

# Example PDP for variable "SPEED_ZONE"
pdp_speed <- partial(
  object = final_model,
  pred.var = "SPEED_ZONE",
  train = X_df,
  type = "classification",  # for binary classification
  prob = TRUE               # get probabilities
)
plotPartial(pdp_speed, main = "Partial Dependence: SPEED_ZONE")


#Example PDP for variable "AGE"
#Example PDP for variable "hour"
#--------------------------------------------------------------
# Interpretability with ICE plots 
#--------------------------------------------------------------

set.seed(34)
iceID <- sample(1:nrow(X_df), 200) #Just plot 200 to avoid overplotting


#Example ICE for variable "SPEED_ZONE"
ice_speed <- partial(
  object = final_model,
  pred.var = "SPEED_ZONE",
  train = X_df[iceID, ],
  type = "classification",  # for binary classification
  prob = TRUE,               # get probabilities
  ice = TRUE
)
plotPartial(ice_speed)

#Example ICE for variable "AGE"
#Example ICE for variable "hour"
#--------------------------------------------------------------
# SHAP values
#--------------------------------------------------------------

#Compute shap values
shap_result <- shap.values(
  xgb_model = final_model,
  X_train = X_full)

#get shap values in long format
shap_result_long <- shap.prep(shap_contrib = shap_result$shap_score,
                              X_train = X_full, top_n = 20) 

#Plot SHAP values for the cases with highest and lowest predicted probability
predTrainxgboost <- predict(final_model, newdata = dtrain)
maxID <- which.max(predTrainxgboost) #Highest predicted probability
minID <- which.min(predTrainxgboost) #Lowes predicted probability

predTrainxgboost[minID]
predTrainxgboost[maxID]


#Create a data frame for plotting
ID_plot <- c(minID, maxID)  #Cases to plot
shap_data_plot <- shap_result_long %>% filter(ID %in% ID_plot)

ggplot(shap_data_plot) + 
  geom_bar(aes(x = reorder(variable, abs(value)), 
                                          y = value),
                                      stat = "identity") + 
  labs(x = "", y = "Feature value contribution", title = "Accidents with highest and lowest probability") +
  coord_flip() + 
  facet_wrap(~ID) +
  theme_bw()

#Create a table with the values for the cases with highest and lowest predicted probability

outTable <- t(vicDataTrain[ID_plot, ] %>% select(SPEED_ZONE, ACCIDENT_TYPE, hour, AGE, SEX))

outDF <- as.data.frame(outTable) %>% rownames_to_column()

colnames(outDF) <- c("Variable", "Min", "Max")

outDF

