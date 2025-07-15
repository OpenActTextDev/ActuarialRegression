#--------------------------------------------------------------
# Packages 
#--------------------------------------------------------------

library(tidyverse) # For data manipulation and visualization
#install.packages("glmnet") 
library(glmnet)    # For Lasso and Ridge regression
library(rpart)     # For fitting decision trees
library(rpart.plot) # For plotting decision trees
#install.packages("pROC")
library(pROC)      # For ROC curves and model evaluation

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

head(vicDataPre) # check the data after removing variables


#Create new variables based on data and time

vicDataPre <- vicDataPre %>% 
  mutate(hour = (hour(ACCIDENTTIME)), 
         ACCIDENTDATE = as.Date(ACCIDENTDATE, format = "%d/%m/%Y"),
         month = (month(ACCIDENTDATE)),
         year = (year(ACCIDENTDATE)),
         hour_fac = factor(hour),       #Use factors for glm but continuous for 
         month_fac = factor(month),     #for tree-based methods
         year_fac = factor(year)) %>% 
  rename(AGE_GROUP = `Age Group`) %>% 
  select(-ACCIDENTTIME, -ACCIDENTDATE)

#Convert character variables to factor
vicDataPre[] <- lapply(vicDataPre, function(x) if(is.character(x)) as.factor(x) else x)


#--------------------------------------------------------------
# Exploratory plots
#--------------------------------------------------------------

#Plot by hour 
fat_hour <- vicDataPre %>% 
  group_by(hour_fac, SEX) %>% 
  summarise(rate = mean(fatal_cnt))

ggplot(fat_hour %>% filter(SEX != "U")) + 
  geom_line(aes(x = hour_fac, y = rate, group = SEX, colour = SEX)) +
  labs(title = "Fatality rate by hour", x = "hour")

#Rate by seatbelt and helmet use
vicDataPre %>% 
  group_by(HELMET_BELT_WORN) %>% 
  summarise(rate = mean(fatal_cnt))


#--------------------------------------------------------------
# Split the data into training and testing
#--------------------------------------------------------------

n <- nrow(vicDataPre)
set.seed(123)

indexTrain <- sample(1:n, round(n*0.8)) #Do 80/20 split

vicDataTrain <- vicDataPre[indexTrain, ] 
vicDataTest <- vicDataPre[-indexTrain, ] 

#--------------------------------------------------------------
# GLM
#--------------------------------------------------------------

#Simple glm using only few variables

glmSimple <- glm(fatal_cnt ~ SEX +  AGE_GROUP + HELMET_BELT_WORN + 
                   Weekday + hour_fac, family = binomial(),
                 data = vicDataTrain)

summary(glmSimple)

#plot coef by hour
coef_glm <- glmSimple$coefficients 

hour_coef <- c(0, coef_glm[grep("^hour_fac", names(coef_glm))])

plot(x = 0:23, y = hour_coef, type = "l", xlab = "hour", 
     ylab = "coefficient", main = "Hour coefficients from GLM") 

#Predicted probabilities in train and test for glm
predTest <- data.frame(glm = predict(glmSimple, newdata = vicDataTest, type = "response"))
predTrain <- data.frame(glm = fitted(glmSimple, type = "response"))


#--------------------------------------------
# Model evaluation
#--------------------------------------------

ROC_test_glm <- roc(vicDataTest$fatal_cnt, predTest$glm)
# Plot ROC curves
plot(ROC_test_glm, col = "red", main = "ROC Curves for GLM", print.auc = TRUE)
#generate AUC
auc(ROC_test_glm)


#--------------------------------------------------------------
# Regularised GLM (Lasso) 
#--------------------------------------------------------------

#Get design Matrix
X <- model.matrix(fatal_cnt ~ ., data = vicDataTrain)[,-1] 
Y <- vicDataTrain$fatal_cnt
lasso <- glmnet(x = X, y = Y, alpha = 1, family = "binomial")

#Plot regularisation path
plot(lasso, xvar = "lambda")  

#Cross-validation to select lambda
set.seed(2)
cv.lasso <- cv.glmnet(x = X, y = Y, family = "binomial",
                      alpha = 1, 
                      nfolds = 5, #Use 5 folds to reduce computation time
                      type.measure = "auc")
#Plot cross-validation results
plot(cv.lasso)
#Get the best lambda
best_lambda <- cv.lasso$lambda.min
#Fit the model with the best lambda
lassoModel <- glmnet(x = X, y = Y, alpha = 1, family = "binomial", 
                     lambda = best_lambda)


#predict on test data
newX <- model.matrix(fatal_cnt ~ ., data = vicDataTest)[,-1]
predTest$lasso <- predict(lassoModel, newx = newX, type = "response")

#Evaluate the model on test data
ROC_test_lasso <- roc(vicDataTest$fatal_cnt, as.numeric(predTest$lasso))
# Plot ROC curves
plot(ROC_test_lasso, col = "blue", main = "ROC Curves for Lasso", print.auc = TRUE)
#generate AUC
auc(ROC_test_lasso)

#------------------------------------------------
# Tree
#------------------------------------------------

library(rpart)
library(rpart.plot)

#set control
control <- rpart.control(minsplit = 10, #Minimum number of splits
                         maxdepth = 3,  
                         minbucket = 100,   
                         cp = -1)

rpartModel <- rpart(fatal_cnt ~ SEX +  AGE + HELMET_BELT_WORN + 
                      Weekday + hour, data = vicDataTrain, method = "class", 
                    control = control) 

rpart.plot(rpartModel, type = 2, digits = 3)


#Predicted probabilities in train and test for tree

predTestTree <- predict(rpartModel, newdata = vicDataTest, type = "prob")
head(predTestTree)
predTest$tree <- predTestTree[,2] #Get the probability of fatality
#Evaluate the model on test data
ROC_test_tree <- roc(vicDataTest$fatal_cnt, predTest$tree)
# Plot ROC curves
plot(ROC_test_tree, col = "green", main = "ROC Curves for Tree", print.auc = TRUE)
#generate AUC
auc(ROC_test_tree)

#--------------------------------------------------------------
# Compare models
#--------------------------------------------------------------

roc_list <- list(GLM = ROC_test_glm, Lasso = ROC_test_lasso, Tree = ROC_test_tree)
roc_df <- data.frame(
  Model = names(roc_list),
  AUC = sapply(roc_list, auc)
)
roc_df
#Plot ROC curves for all models
plot(roc_list$GLM, col = "red", main = "ROC Curves for All Models")
plot(roc_list$Lasso, col = "blue", add = TRUE)
plot(roc_list$Tree, col = "green", add = TRUE)
legend("bottomright", legend = names(roc_list), col = c("red", "blue", "green"), lwd = 2)
