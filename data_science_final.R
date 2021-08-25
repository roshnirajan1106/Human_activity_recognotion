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
data$new_window <- as.factor(data$new_window)
#data$classe

data$classe <- as.factor(data$classe)
#data$new_window



realTesting <- read.table("pml-testing.csv", sep = ",", header=T, na.strings = c("#DIV/0!", "", " ", "NA"))
realTesting[is.na(realTesting)] <- 0
realTesting <- realTesting[,c(validColumns,-1,-2)] # Remove all columns with >=95% of zeros

# Adaptations to realTesting set in order to match types of training dataset.
colnames(realTesting)[colnames(realTesting)=="problem_id"] <- "classe"
realTesting$classe <- as.factor(realTesting$classe)
realTesting$classe <- as.factor("A")
realTesting$magnet_forearm_y <- as.numeric(realTesting$magnet_forearm_y)
realTesting$magnet_forearm_z <- as.numeric(realTesting$magnet_forearm_z)
realTesting$magnet_dumbbell_z <- as.numeric(realTesting$magnet_dumbbell_z)








#shuffle the data

set.seed(123)
data<-data[order(runif(19622)),]




correlations <- cor(data[sapply(data, is.numeric)], use='pairwise')
corrplot(correlations,method = "circle", tl.cex = 0.54, tl.col = 'black', order = "hclust", addrect = 5)



data <- subset(data, select = -c(raw_timestamp_part_1,cvtd_timestamp,new_window,pitch_forearm,raw_timestamp_part_2,total_accel_forearm,roll_forearm,num_window))
cat("The dimension of the data after cleaning / removing least correlated vairables : ", dim(data))

realTesting <- subset(realTesting, select = -c(raw_timestamp_part_1,cvtd_timestamp,new_window,pitch_forearm,raw_timestamp_part_2,total_accel_forearm,roll_forearm,num_window))


boxplot( magnet_belt_y ~ classe, data = data, xlab = "classe",
         ylab = "magnet_belt_y",col = c("blue","red","yellow","green","pink"), main = "Data Visualization")

boxplot( gyros_arm_x ~ classe, data = data, xlab = "classe",
         ylab = "gyros_arm_x",col = c("blue","red","yellow","green","pink"), main = "Data Visualization")

boxplot( yaw_arm ~ classe, data = data, xlab = "classe",
         ylab = "yaw_arm",col = c("blue","red","yellow","green","pink"), main = "Data Visualization")

boxplot( yaw_dumbbell ~ classe, data = data, xlab = "classe",
         ylab = "yaw_dumbbell",col = c("blue","red","yellow","green","pink"), main = "Data Visualization")

boxplot( accel_arm_y ~ classe, data = data, xlab = "classe",
         ylab = "accel_arm_y",col = c("blue","red","yellow","green","pink"), main = "Data Visualization")



#Normalization 

n2<-function(b){
  
  (b-min(b))/(max(b)-min(b))
}
data_n <- data[, 1:49]

normalized_data <- as.data.frame(lapply(data_n, n2))
data_n <- normalized_data

test_n <- realTesting[,1:49]

normalized_testing <- as.data.frame(lapply(test_n, n2))

test_n <- normalized_testing
lable <- realTesting[,50]
test_n <- cbind(lable , test_n)
View(test_n)



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
a<- sum(data_test$lable == "A")
a







qplot(magnet_forearm_x,magnet_forearm_y, colour=lable, data=data_train)
qplot(gyros_dumbbell_x,gyros_dumbbell_y, colour=lable, data=data_train)
qplot(accel_forearm_x,accel_forearm_y, colour=lable, data=data_train)
qplot(gyros_forearm_x,yaw_forearm, colour=lable, data=data_train)







train.control <- trainControl(method = "cv", number = 10)
classifier_cl <- naiveBayes(lable ~ . ,data_train,trControl = train.control )
y_pred <- predict(classifier_cl, newdata = data_test)
ct <- table(y_pred, data_test[, 1])
ct


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

############################################













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


pre_rfF<-predict(random,test_n ,type = "response")
pre_rfF


