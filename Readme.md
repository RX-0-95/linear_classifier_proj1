# update 
The classifer should be able to handle multiclass now, and the f function will directly return the class of the predict class now. 
See example below 

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

            #Train 
            classifier = MyClassifier(2, 28*28)
            classifier.train(train_data, train_label) 

            #Test
            correct_count = 0
            wrong_count = 0 
            for i, data in enumerate(Test_data):
                predict_class = classifier.f(data)
                if(predict_class != Test_label[i]):
                    wrong_count +=1
                else:
                    correct_count +=1 


## Test Result 
* I haven't run test with the train data set because it will take amlost an hour to run on my pc. 
  But I did tested for 4 classes of test data set (1 7 2 8), the code generate 6 classifiers 
  and reach 100% accuracy because I use test data to train the model. 