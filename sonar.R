install.packages("mlbench")
install.packages("caTools")
install.packages("ROCR")
install.packages("glmnet")
install.packages("DataExplorer")
library(glmnet)
library(mlbench)
library(caTools)
library(ROCR) 
library(DataExplorer)

# Data exploration
data(Sonar)
dim(Sonar)
DataExplorer::create_report(Sonar)

Sonar[1:6, c(1:5, 61)]

# splitting data
rows <- sample(nrow(Sonar))
Sonar <- Sonar[rows, ]
split <- round(nrow(Sonar) * .60)
train <- Sonar[1:split, ]
test <- Sonar[(split + 1):nrow(Sonar), ]
nrow(train) / nrow(Sonar)
summary(Sonar)

col_len <- ncol(train)

x_train <- train[,1:col_len-1]
y_train <- train[,col_len]


x_test <- test[,1:col_len-1]
y_test <- test[,col_len]

summary(y_train)
summary(y_test)
dim(x_train)
dim(x_test)


# Fit the model
glmmod <- glmnet(x=as.matrix(x_train), y=as.factor(y_train), alpha=1, family="binomial")
coef(glmmod)
plot(glmmod, xvar="lambda")

# Predict test data based on model
predict_reg <- predict(glmmod, as.matrix(x_test), type = "response")
predict_reg

# Changing probabilities
predict_reg <- ifelse(predict_reg >0.5, 1, 0)

dim(predict_reg)
dim(as.matrix(y_test))
# Evaluating model accuracy
# using confusion matrix
table(as.matrix(y_test), predict_reg)

missing_classerr <- mean(predict_reg != test_reg$vs)
print(paste('Accuracy =', 1 - missing_classerr))
# 
# # ROC-AUC Curve
# ROCPred <- prediction(predict_reg, test_reg$vs) 
# ROCPer <- performance(ROCPred, measure = "tpr", 
#                       x.measure = "fpr")
# 
# auc <- performance(ROCPred, measure = "auc")
# auc <- auc@y.values[[1]]
# auc
# 
# # Plotting curve
# plot(ROCPer)
# plot(ROCPer, colorize = TRUE, 
#      print.cutoffs.at = seq(0.1, by = 0.1), 
#      main = "ROC CURVE")
# abline(a = 0, b = 1)
# 
# auc <- round(auc, 4)
# legend(.6, .4, auc, title = "AUC", cex = 1)
