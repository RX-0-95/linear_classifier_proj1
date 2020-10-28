# Instruction on How to use Train and f function in classifer 
The code for the project is inside proj_linear_classifer_notebk.ipynb file, rest of the files are sketches 

## Data Set 
The data set are mnist_test.csv and mnist_train.csv file, the link is provided in the project file too.  
link: https://www.kaggle.com/oddrationale/mnist-in-csv

## Dependence 
pandas 
numpy 
cvxpy 
matplotlib (Just for testing, will not be include in the final version)

## Train and f function 
The helper function I wrote for extract 2 classes from the train set and test set is used in this example. 

        classifier = MyClassifier(2, 28*28) #Create the classifier object 
        Train_data_17,Train_label_17 = extract_two_digit(1,7,Train_data,Train_label) # This function extract data set only contain 1 and 7 from the entire data set
        Test_data_17,Test_label_17 = extract_two_digit(1,7,Test_data,Test_label) # same for the test data set 
        classifier.train(0.6,Train_data_17, Train_label_17) # train the classifier 

        #######################Test the classifier##################################
        #When the classifer is trained, it treat the first data set in the train data set as lable 1. For example, if in the train data set for class 2, 8, the first
        # set is digit 2, than function f() will output +1 if the classifier classify the data to be 2, and -1 if the input data is recognized as 8.  
        
        labelPresent_1 = Train_label_17[0] #The first data in the train data set will 1 in this binary classifer, the other one is -1 
        correct_count=0
        wrong_count = 0 
        for i in range(0,len(Test_label_17)):
            label = 1
            if(Test_label_17[i] != labelPresent_1):
                label = -1 
            if(classifier.f(Test_data_17[i]) != label): #use function f() to predict the input data, and compare to the correct label
                wrong_count+=1
            else:
                correct_count+=1 
        correct_percentage = 100*correct_count/(correct_count+wrong_count)
        print ("correct: ", correct_count)
        print("wrong: ", wrong_count)
        print("Precision: " ,round(correct_percentage,4),"%")


## Test Result 
* p = 0: The test result for (1 7) classifier is 99.2141 % correct in test set. 
* p = 0.6: 99.1216 % correct 