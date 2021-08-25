# Human_activity_recognition



1.	Project Title:
HUMAN ACTIVITY RECOGINITON
2.	Data Set Name:
Human Activity Recognition dataset

3.	Data set Link:
•	Dataset: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv.

4.	Software used:
R , R studio
5.	Objective:
Human activity recognition  is the problem of predicting what a person is doing based on a trace of their movement using sensors. The objective of this project is to analyze a dataset which contains the data of the sensors which records data like accerlation, gyro, gravity accerlation in all three directions and many more of the body which is attached to the smartphone and helps draw insights and predict the activity using Machine Learning .
6.	Project Description:

a.	Dataset and Loading  - This data has been collected from an implementation of an on-body sensing approach , in which the individual is equipped with 4  accelerometers/gyroscopes/magnetometers sets, 3 of them on-body and 1 in an exercising tool. They measure: Arm orientation, Forearm orientation, Belt orientation and Dumbbell orientation.

b.	Data cleaning and Data preprocessing  - After loading the dataset from the link mentioned above . It contains about 19622   rows and 160 variables . This data set had plenty of irregularities like empty elements, division by zero, blank spaces and NAs . We filled this null and insignificant values with 0 . We then removed all the columns which consisted of more than equal to 95% of null values. This resulted in 19622 variables and 58 variables .Further we found out the correlation between each variable and cleaned or removed those which had the correlation coefficient from -0.2 to  0.2 .  This resulted in 19622 rows and 50 variables .

c.	Data visualization – In our project we’ve used box plot , qqplots between randomly picked variables and the labeled variable or the class  only to check the degree of separability. The corrplot() function from corrplot library was used to visualize correlations among different variables . Lastly, we’ve used ggplot to plot the relation between the random forest and the error rate.


d.	Cross Validation – We’ve cross validated our data which was split into training and testing dataset . Our dataset is very noisy thus we got an overfitting model with low bias and high variance . Also it’s not a good practice in general to stick only a subset of dataset . So for the same reason we used k- fold cross validation. Typically  one performs k-fold cross-validation using k = 5 or k = 10, as these values shows test error rate  that suffer neither from excessively high bias nor from very high variance.

e.	Modelling -  We’ve tried our hand on 3 classification algorithm. The first one that we’ve used is naïve bayes classifier as our first classification theorem . Then we fitted our training dataset in KNN model with K=5 . Then we choose to fit the same dataset again in random forest model . We’ve chosen the model on the basis of accuracy , we’ve selected that which gives the best accuracy

f.	Data and it’s output :
For the making of this dataset six participants were involved to perform 10 times  Dumbbell biceps curls . The measurement is taken from accelerometers, gyro sensor , magnetometer .
The output factor levels are :
A: Throwing the hips to the Back.
B: Throwing the elbow to the front.
C: lifting the dumbbell halfway up.
D: lowering the dumbbell halfway.
E: Throwing the hips to the front.

Libraries used :

1. ggplot2:  For various visualization purpose
2. corrplot : To draw or visualize the correlation between the variables
3. caret: It contains functions to streamline the model training process for complex regression and classification problems

4. e1071 : for naïve bayes
5. randomForest: for building the random forest model



Algorithm Used :
•	Naïve bayes
•	K nearest Neighbor
•	Random Forest


g.	Code:
library(randomForest)
library(gmodels)
library(ggplot2)
library(corrplot)
library(e1071)
library(caTools)
library(class)
library(caret)
set.seed(123)

data <- read.table("pml-training.csv", sep = ",", header=T, na.strings = c("#DIV/0!", "", " ", "NA"))



data[is.na(data)] <- 0
data$classe

validColumns <- -which(as.numeric(colSums(data==0)) >= nrow(data)*0.95) # columns with quantity of zeros < percentage
#removing  those columns where more than 95% of the data is 0
#removing the first and second column
cat("The original dimension of the data : ", dim(data))
data <- data[, c(validColumns,-1,-2)]
cat("The dimension of the data after null values : ", dim(data))

#data$classe

data$classe <- as.factor(data$classe)
#data$new_window

#shuffle the data

set.seed(123)
data<-data[order(runif(19622)),]

correlations <- cor(data[sapply(data, is.numeric)], use='pairwise')
corrplot(correlations,method = "circle", tl.cex = 0.54, tl.col = 'black', order = "hclust", addrect = 5)


The coorelation plot to find the correlation between each variables

data <- subset(data, select = -c(raw_timestamp_part_1,cvtd_timestamp,new_window,pitch_forearm,raw_timestamp_part_2,total_accel_forearm,roll_forearm,num_window))
cat("The dimension of the data after cleaning / removing least correlated vairables : ", dim(data))

boxplot( magnet_belt_y ~ classe, data = data, xlab = "classe",
ylab = "magnet_belt_y",col = c("blue","red","yellow","green","pink"), main = "Data Visualization")

boxplot( gyros_arm_x ~ classe, data = data, xlab = "classe",
ylab = "gyros_arm_x",col = c("blue","red","yellow","green","pink"), main = "Data Visualization")

boxplot( yaw_arm ~ classe, data = data, xlab = "classe",
ylab = "yaw_arm",col = c("blue","red","yellow","green","pink"), main = "Data Visualization")

boxplot( yaw_dumbbell ~ classe, data = data, xlab = "classe",
ylab = "yaw_dumbbell",col = c("blue","red","yellow","green","pink"), main = "Data Visualization")

boxplot( accel_arm_y ~ classe, data = data, xlab = "classe",
ylab = "accel_arm_y",col = c("blue","red","yellow","green","pink"), main = "Data

Visualization")
#Normalization
n2<-function(b){

(b-min(b))/(max(b)-min(b))
}
data_n <- data[, 1:49]

normalized_data <- as.data.frame(lapply(data_n, n2))
data_n <- normalized_data
#View(data_n)
training = data_n[ 1:13500 , ]
testing = data_n[13501:19622, ]
train_lable <- data[1:13500,50]
test_lable <- data[13501:19622,50]
lable = train_lable
data_train <- cbind(lable,training)
lable <- test_lable
data_test <- cbind(lable, testing)
data_new <- rbind(data_train, data_test)
qplot(magnet_forearm_x,magnet_forearm_y, colour=lable,
data=data_train)
qplot(gyros_dumbbell_x,gyros_dumbbell_y, colour=lable, data=data_train)
qplot(accel_forearm_x,accel_forearm_y, colour=lable, data=data_train)
qplot(gyros_forearm_x,yaw_forearm, colour=lable, data=data_train)



The above plots was to take a look at the dataset , for example how they are clustered and the presence of outliers.
From the above plots we can observe that the data is congested and we can see very less amount of outliers

train.control <- trainControl(method = "cv", number = 10)
classifier_cl <- naiveBayes(lable ~ . ,data_train,trControl = train.control )
y_pred <- predict(classifier_cl, newdata = data_test)
ct <- table(y_pred, data_test[, 1])
ct
acc<-sum(diag(ct))/sum(ct)
acc



#####################KNN
train.control <- trainControl(method = "cv", number = 10)
model <- train(lable ~., data = data_train, method = "knn",
trControl = train.control)
pre_rfF<-predict(model,data_test )
ct <- table(pre_rfF, data_test[, 1])
acc<-sum(diag(ct))/sum(ct)
prec <- 1708 / (1708 + 19+ 10+6)
prec
confusionMatrix(pre_rfF, data_test[, 1])
cat("Accuracy with KNN : ",acc)


#########random forest ######
random<-randomForest(lable ~ . ,data_train )
print(random)
pre_rfF<-predict(random,data_test ,type = "response")
ct <- table(pre_rfF, data_test[, 1])
acc<-sum(diag(ct))/sum(ct)
confusionMatrix(pre_rfF, data_test[, 1])

cat("Accuracy with random forrest : ",acc)
pre_rfF<-predict(random,test_n ,type = "response")
pre_rfF
error_data <- data.frame(
Trees = rep(1:nrow(random$err.rate) ,times = 5),
Type = rep(c("A","B","C","D","E"), each = nrow(model$err.rate)),
Error = c(random$err.rate[,"A"],
random$err.rate[,"B"],
random$err.rate[,"C"],
random$err.rate[,"D"],
random$err.rate[,"E"]))
g <- ggplot(data = error_data,size = 50, aes(x=Trees, y= Error))
g<- g +  geom_line(aes(color =Type))
g <- g+ ggtitle("Error Rate vs Trees used in rf")
#g <- g+ theme(plot.title = element_text(size=30, face="bold", vjust=1,mar lineheight=0.6))
g

The above plot mentions the error rate vs the number of trees taken in random forest – from this plot we can see that the error rate becomes constant  at a point , from that point even if increase the number of trees used in random forest it won’t have any significant change



For Naïve bayes
























For knn :



















For random forest :







7.	Results: Quantitative findings and Plots.
•	Naïve bayes – Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems. ... Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions. We got an accuracy of 45 % in naïve bayes

•	Knn -  we first cross validated our data set with the method k – fold and kept k= 5 which gives less error. Before the training of this model the dataset is normalized . The parameter k which is nothing but the first K nearest neighbor . Here K=5 gives less error compared to othe values and this model gives an accuracy of 95% .
Now the confusion matrix is drawn and parameters like accuracy, precision, recall and specificity are calculated. Which is comparatively
greater than naïve bayes .

•	Random forest -  After Knn we’ve used random forest and around 500 number of tress are used and oob is estimated.  It gives an accuracy of 99.5% and thus stands out among all the classifier . We have not used cross validation for building random forest model , first reason is the difference between cross validated model and non-cross validated model has insignificant difference , it is very less , and the second reason is because we’ve around 13k rows and 50 variables in training dataset , and 500 trees are generated so it was taking a large amount of time and computation power so we dropped the cross validated model .

	Naïve Bayes	KNN	Random Forest
Accuracy 	0.452	 0.9502	0.9931
Kappa 	0.328	0.937	0.9913
Other significant values : 		K=5 gives the best accuracy 	500 decision  trees were used build the random Forest model and gave the best accuracy







8.	Conclusions:
This project was to predict human activity behavior using sensors .We can see now a days we’ve smartwatches , fitness watch etc. which helps us know how many steps we’ve taken , our heart rate etc. . Sensors generate a huge amount of data per sec , so it is very important to process it, clean it before fitting to a model which helps in predicting the activity of the person . So in our project we’ve used three classifier algorithm – Naïve bayes, knn , random forest which gives an accuracy of 45%, 95%, 99% respectively and thus concluding that random forest is most obvious choice for predicting the human behavior for the given dataset . Also as a part of future scope This project can be implemented in workspace , schools for  activity detection, monitoring persons for signs of fatigue, distinguishing one individual from another , with possible deployment in highly sensitive and secure workplaces etc.
