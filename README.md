# Sparkify
## *A capstone project of the [datascience nanodegree by Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025)*

### Project overview
This is the final project of the datascience nanodegree (DSND). We have an example of a virtual company called 'Sparkify' who offers paid and free listening service, the customers can switch between either service, and they can cancel their subscription at any time.
The given customers dataset is huge (12GB), thus the standard tools for analysis and machine learning will not be useful here as it will not fit in the computer's memory (even the data can fit in the memory of a 32GB computer, the analysis, and computations require way more than that amount, and the analysis will crash the system). The safe way to perform such analysis is to do it using Big Data tools like Apache Spark which is one of the fastest big data tools.
> For this tutorial, we worked with only a 128MB slice of the original dataset.

**The problem** is to create a machine learning model to predict the user intent to unsubscribe (Called customers' churn) before this happens.

### Files in this repo

* **Folders**
  * [BlogPics](https://github.com/drnesr/Sparkify/tree/master/BlogPics): A folder contains the pictures used in the blog post.
  * [Trained_models](https://github.com/drnesr/Sparkify/tree/master/Trained_models): A folder contains the trained models.		
    * DecisionTreeClassifier.model
    * GradientBoostedTrees.model
    * LogisticRegression.model
    * MultilayerPerceptronClassifier.model
    * RandomForestClassifier.model
    * model_LogReg.model
	
* **Files**
  * [Sparkify.ipynb](https://github.com/drnesr/Sparkify/blob/master/Sparkify.ipynb): The main coding file in jypyter notebook format to work in Udacity workspace.
  * [GeneralizeSparkify.py](https://github.com/drnesr/Sparkify/blob/master/GeneralizeSparkify.py): The main code to allpy the generalize the procedures to any dataset
  * [Sparkify.html](https://github.com/drnesr/Sparkify/blob/master/Sparkify.html): an HTML file for the jupyter workbook
  * [README.md](https://github.com/drnesr/Sparkify/blob/master/README.md): This file.
  * [saved_user_dataset.CSV](https://github.com/drnesr/Sparkify/tree/master/saved_user_dataset.CSV): The extracted dataset for machine learning in spark format
  * [saved_user_dataset_pd.CSV](https://github.com/drnesr/Sparkify/blob/master/saved_user_dataset_pd.CSV): The extracted dataset for machine learning in csv format
  * [.gitignore](https://github.com/drnesr/Sparkify/blob/master/.gitignore): GIT ignore file (files and folder to be ignored)
  * [@JupyterHere.bat](https://github.com/drnesr/Sparkify/blob/master/%40JupyterHere.bat): A batch file to run jupyter notebook here in one click (Windows only)
  * Sparkify_NB.ipynb: The main coding file in jypyter notebook format to work in IBM Watson workspace (Not complete).



### Files NOT in this repo
* mini_sparkify_event_data.json: A 128MB json file contains the main data (exceeds GitHub limit)

### Used resources
- Python 3.6.x (The programing language)
- pySpark 2.4.x (Machine learning library for big data)
- matplotlib 3.03 (A plotting library)
- pandas 0.23 (numerical calculations library)
- jupyter (The programming notebook interface)

### Generalization
Through the file [`GeneralizeSparkify.py`](https://github.com/drnesr/Sparkify/blob/master/GeneralizeSparkify.py), we can do the following:

**First, import Stage#1 command**

```from GeneralizeSparkify import load_clean_transfer```

**Assume we have new data named `'new_data.json'`, write this command:**

```load_clean_transfer('new_data.json', save_as='new_dat_extraction')```

This command will:
1. read the data from the given source,
2. clean the data
3. save the cleaned dataset to the new name as `new_dat_extraction.CSV`

**Next, import Stage#2 commands**

```from GeneralizeSparkify import load_ml_dataset, get_train_test_features, apply_model```

**2.1. We should load the saved extracted data `new_dat_extraction.CSV`**

```ml_ds = load_ml_dataset(saved_as='new_dat_extraction.CSV')```

**2.2 Then get the train, test portions and the features names**

```train, test, features_labels = get_train_test_features(ml_ds)```

***2.3 Finally, apply an ML model to the data***

***Either by creating the model:***

```apply_model(train, test, features_labels, model_name='GBT',save_as='NewGBT.model')```

***Or by loading existing model***

```apply_model(train, test, features_labels,model_name='LR', load_from_existing='LogisticRegression.model')```

### More details
Please refer to [This medium post](https://medium.com/@drnesr/dont-let-them-go-818c03d7f09e) for details about this model.
