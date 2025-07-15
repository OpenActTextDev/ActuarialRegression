#--------------------------------------------------------------
# Packages 
#--------------------------------------------------------------

library(tidyverse) # For data manipulation and visualization
install.packages("glmnet") #This may not be installed in your R environment
library(glmnet) #For lasso and ridge regression

#--------------------------------------------------------------
# Prepare data
#--------------------------------------------------------------

KCdata <- read_csv("https://raw.githubusercontent.com/OpenActTextDev/ActuarialRegression/refs/heads/main/CourseCSVData/kc_house_data.csv") 

str(KCdata) # Check the structure of the data

# Remove unnecessary columns
KCdata <- KCdata %>%
  select(-id, -date, -zipcode, -lat, -long,
         -grade, 
         -yr_renovated, #many houses have not been renovated
         -sqft_basement,) #sqft basement as it is redundant with sqft_living
# Convert variables to factors
KCdata <- KCdata %>%
  mutate(
    waterfront = as.factor(waterfront),
    view = as.factor(view),   #Index (0–4) of how good the view is
    condition = as.factor(condition) # Index (1–5) of the condition of the house
  )



#----------------------------------------------------------------------
# Descriptive Analysis
#-----------------------------------------------------------------------


#Price vs size of the house
ggplot(KCdata) + geom_point(aes(x = sqft_living, y = price)) 

#Price vs number of bedrooms
ggplot(KCdata) + geom_point(aes(x = bathrooms, y = price)) 

#Price vs number of bedrooms
ggplot(KCdata, aes(x = waterfront, y = price)) + geom_boxplot() 


#--------------------------------------------------------------
# Split the data into training and testing
#--------------------------------------------------------------

set.seed(123) # Set seed for reproducibility

n <- nrow(KCdata)
indexTrain <- sample(1:n, round(n*0.8)) #Do 80/20 split
KCdataTrain <- KCdata[indexTrain, ]
KCdataTest <- KCdata[-indexTrain, ]

# Check the number of rows in the training set
n
length(indexTrain)

#----------------------------------------------------------------------
# Linear regression
#-----------------------------------------------------------------------

#Fit
lmArea <- lm(price ~ sqft_living, data = KCdataTrain)
#Note that we are only using sqft_living as the predictor variable and using the
#training data set KCdataTrain

# Print the summary of the linear model
summary(lmArea)

#----------------------------------------------------------------------
# Multiple Linear regression
#-----------------------------------------------------------------------

#Fit
lmAll <- lm(price ~ ., data = KCdataTrain)
# Note the use of '.' to include all other variables in the dataset
#Alternatively, you can specify each variable explicitly
summary(lmAll)


#----------------------------------------------------------------------
# Predictions and Evaluation
#-----------------------------------------------------------------------

# Calculate the Residual Sum of Squares (RSS) for both models on training and test data

#Predictions on training data
yhat_lmArea <- fitted(lmArea)
lmAreaRSS_tr <- mean((yhat_lmArea - KCdataTrain$price)^2)
yhat_lmAll <- fitted(lmAll)
lmAllRSS_tr <- mean((yhat_lmAll - KCdataTrain$price)^2)

#Predictions on test data
yhat_lmArea_test <- predict(lmArea, newdata = KCdataTest)
lmAreaRSS_te <- mean((yhat_lmArea_test - KCdataTest$price)^2)
yhat_lmAll_test <- predict(lmAll, newdata = KCdataTest)
lmAllRSS_te <- mean((yhat_lmAll_test - KCdataTest$price)^2)

# Print the RSS for both models
lmAreaRSS_tr
lmAreaRSS_te
lmAllRSS_tr
lmAllRSS_te


#----------------------------------------------------------------------
# Ridge and lasso regression
#-----------------------------------------------------------------------

#We first to create a model matrix for the predictors
X <- model.matrix(lmAll)[,c(-1)] # Exclude the intercept column as it is not needed for glmnet: it is automatically included


# Create the response variable
Y <- KCdataTrain$price

# Fit a ridge regression model

ridge_model <- glmnet(X, Y, alpha = 0) # alpha = 0 for ridge regression
# Fit a lasso regression model
lasso_model <- glmnet(X, Y, alpha = 1) # alpha = 1 for lasso regression
# Plot the coefficients for ridge regression
plot(ridge_model, xvar = "lambda", label = TRUE, 
     main = "Ridge Regression Coefficients")
# Plot the coefficients for lasso regression
plot(lasso_model, xvar = "lambda", label = TRUE, 
     main = "Lasso Regression Coefficients")

# Perform cross-validation to find the optimal lambda for ridge regression
cv_ridge <- cv.glmnet(X, Y, alpha = 0, type.measure = "mse") 
#Note the number of folds is set to 10 by default. It can parameter changed by setting the `nfolds` argument

# Perform cross-validation to find the optimal lambda for lasso regression
cv_lasso <- cv.glmnet(X, Y, alpha = 1, type.measure = "mse")

# Plot the cross-validation results 
plot(cv_ridge)
plot(cv_lasso)
# Get the optimal lambda values
ridge_lambda_opt <- cv_ridge$lambda.min
lasso_lambda_opt <- cv_lasso$lambda.min
# Print the optimal lambda values
ridge_lambda_opt
lasso_lambda_opt

# Fit the final ridge and lasso models using the optimal lambda values
ridge_final <- glmnet(X, Y, alpha = 0, lambda = ridge_lambda_opt)
lasso_final <- glmnet(X, Y, alpha = 1, lambda = lasso_lambda_opt)



# Predictions on the test data using the final ridge model

newx_test <- model.matrix(price ~ ., data = KCdataTest)[, -1] # Create model matrix for test data

yhat_ridge_test <- predict(ridge_final, newx = newx_test)
# Predictions on the test data using the final lasso model
yhat_lasso_test <- predict(lasso_final, newx = newx_test)

# Calculate the RSS for the ridge and lasso models on the test data
ridgeRSS_te <- mean((yhat_ridge_test - KCdataTest$price)^2)
lassoRSS_te <- mean((yhat_lasso_test - KCdataTest$price)^2)

# Print the RSS for the ridge and lasso models
ridgeRSS_te
lassoRSS_te
# Compare the RSS values for all models

cat("RSS for Linear Model with Area:", lmAreaRSS_te, "\n")
cat("RSS for Multiple Linear Model:", lmAllRSS_te, "\n")
cat("RSS for Ridge Regression:", ridgeRSS_te, "\n")
cat("RSS for Lasso Regression:", lassoRSS_te, "\n")
