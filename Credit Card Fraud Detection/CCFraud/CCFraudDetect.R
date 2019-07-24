#Installing Packages
list.of.packages <- c("data.table","caret","ggplot2","heatmaply","corrplot","kernlab","e1071",
                      "MASS","neuralnet","dplyr","hydroGOF")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages,dependencies = TRUE)

#Loading Packages
invisible(lapply(list.of.packages, library, character.only = TRUE))

#Loading Dataset
ccfraud <- read.csv(file='creditcard.csv',header = TRUE,sep = ',')
head(ccfraud)

#Exploratory Data Analysis
fraud_nofraudplot <- table(ccfraud$Class)
fraud_nofraud <- (fraud_nofraudplot/nrow(ccfraud))*100
fraud_nofraud #99.83% of No Frauds in the Dataset & 0.17% Frauds in the Dataset

# convert target variable into factor
ccfraud$Class <- as.factor(ccfraud$Class)

#Scaling the Time variable
ccfraud$Time <- (ccfraud$Time/3600)

#Plotting Fraud Vs Non-Fraud
ggplot(data=ccfraud, aes(x=class)) +
  geom_bar(width=0.2, fill="steelblue") +
  geom_text(stat='count', aes(label=..count..), vjust=-1,color="black", size=4.5) +
  xlab("Class of Fraud") +
  ylab("Count of Transactions") +
  ggtitle("Credit Card Fraud Distribution \n 0: No Fraud | 1: Fraud") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylim(0,300000)

#Correlation Plot
matcor <- cor(ccfraud[,c(1:29)], method = c("pearson", "kendall", "spearman"))
head(round(matcor,2))
corrplot(matcor, method="circle",type="upper",tl.col="black")

#Train-Test Split
set.seed(5)
trainindex <- createDataPartition(ccfraud$Class, p=0.80, list= FALSE)
cc_train <- ccfraud[trainindex, ]
cc_test <- ccfraud[-trainindex, ]

#Logit Model
glm2 <- glm(Class~.,data=cc_train, family = binomial) %>%
  stepAIC(trace = FALSE)
summary(glm2)
pred_glm <- predict(glm2,cc_test,type="response")
pred_glm <- ifelse(pred_glm > 0.5,1,0)
conf.glm <- confusionMatrix(table(Predicted=pred_glm, Actual=cc_test$Class))
#fourfoldplot(conf.glm$table,color = c("#CC6666", "#99CC99"),main="Logit Model Confusion Matrix Plot")
draw_confusion_matrix(conf.glm,'Logit Confusion Matrix')

#Cross Validated Logit Model
train_control = trainControl(method="repeatedcv", number=10, repeats=3)
cross.glm <- train(Class~V1+V4+V5+V6+V8+V9+V10+V13+V14+V16+V20+V21+V22+V23+V27+V28+Amount,
                   data=cc_train,method="glm",family="binomial", trControl=train_control)
summary(cross.glm)
pred.cross.glm <- predict(cross.glm,cc_test,type="raw")
conf.matrix.cross.glm <- confusionMatrix((table(Predicted = pred.cross.glm, Actual = cc_test$Class)))
draw_confusion_matrix(conf.matrix.cross.glm,'K-Fold Cross Validated Logit Confusion Matrix')


#SVM Model
svm.model <- svm(Class ~ ., data = cc_train, kernel = "radial", cost = 10, gamma = 0.1)
summary(svm.model)
svm.predict <- predict(svm.model, cc_test)
svm.conf <- confusionMatrix(table(Predicted=svm.predict, Actual=cc_test$Class))
draw_confusion_matrix(svm.conf,'SVM Model Confusion Matrix')
error.svm <- as.numeric(cc_test$Clas)-as.numeric(svm.predict)
RMSEsvm=rmse(error.svm)



#Neural Network
# nn <- neuralnet(Class~.,data=cc_train, hidden=c(30,25,20,15,10,7,5),act.fct = "logistic",
#              linear.output = FALSE)
# plot(nn)
# nn.results <- predict(nn,cc_test,type="Class")
# conf.nn <- confusionMatrix(table(Predicted=apply(nn.results, 1, which.max)-1, Actual=cc_test$Class))
# fourfoldplot(conf.nn$table,color = c("#CC6666", "#99CC99"),main="Neural Network Confusion Matrix Plot")


nn2 <- neuralnet(Class~.,data=cc_train, hidden=c(20,15,5),act.fct = "logistic",
                linear.output = FALSE)
plot(nn2)
nn2.results <- predict(nn2,cc_test,type="Class")
conf.nn2 <- confusionMatrix(table(Predicted=apply(nn2.results, 1, which.max)-1, Actual=cc_test$Class))
fourfoldplot(conf.nn2$table,color = c("#CC6666", "#99CC99"),main="Neural Network Confusion Matrix Plot")

#***************************************************************
#Function To Plot Confusion Matrix
#***************************************************************

draw_confusion_matrix <- function(cm,titlename) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(titlename, cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, '0', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, '1', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, '0', cex=1.2, srt=90)
  text(140, 335, '1', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

#*******************************************************************
#Function to Calculate RMSE
#*******************************************************************