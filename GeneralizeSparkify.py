
# import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.sql.functions import sum as Fsum

import datetime

from pyspark.sql import Window
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

import re
from pyspark.sql import functions as sF
from pyspark.sql import types as sT

from functools import reduce

# ML imports
from pyspark.ml.feature import Normalizer, StandardScaler, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes, RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# TO load saved models 
from pyspark.ml.tuning import CrossValidatorModel as mlm

# create a Spark session
spark = SparkSession.builder.appName("Sparkify").getOrCreate()

def load_data(data_path):
    # data_path = 'mini_sparkify_event_data.json'
    # Load
    # ----
    df = spark.read.json(data_path)
    return df
    
    
def clean_dataframe(df):
    # Clean
    # -----
    # Dropping the missing User's ID rows
    df = df.filter(df.userId != '')
    # Dropping null session IDs if any
    df = df.filter(df.sessionId != '')
    
    # Adding churn column
    # ...................
    # Define a flag function
    flag_cancelation_event = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
    # apply to the dataframe
    df = df.withColumn("churn", flag_cancelation_event("page"))
    #Define window bounds
    windowval = Window.partitionBy("userId").rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    # Applying the window
    df = df.withColumn("churn", Fsum("churn").over(windowval))
    
    # Adding Time columns
    # ...................    
    # Definig user functions to get hour, day, month, and weekday of cancellation
    get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).hour)
    get_day = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).day)
    get_month = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).month)
    # Source https://stackoverflow.com/questions/38928919/how-to-get-the-weekday-from-day-of-month-using-pyspark
    get_weekday = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime('%w'))    
    
    # Adding columns to the dataframe
    funcs = {'hour':get_hour, 'day':get_day, 'month':get_month, 'week_day':get_weekday}
    for label, func in funcs.items():
        df = df.withColumn(label, func(df.ts))
        print(f'Column {label} added successfully.')
        
    # Adding Operating Systems columns
    # ................................
    # TO get the data between parenthesis
    ex = '\(([^\)]*)\)'
    # Create mappers for the os
    mapper_general = {'Compatible': 'Windows',  'Ipad': 'Mac',  'Iphone': 'Mac',  
              'Macintosh': 'Mac',  'Windows nt 5.1': 'Windows',  
              'Windows nt 6.0': 'Windows',  'Windows nt 6.1': 'Windows',  
              'Windows nt 6.2': 'Windows',  'Windows nt 6.3': 'Windows',  
              'X11': 'Linux'}
    mapper_specific = {'Compatible': 'Windows 7',  'Ipad': 'iPad',  'Iphone': 'iPhone',  
              'Macintosh': 'MacOS',  'Windows nt 5.1': 'Windows XP',  
              'Windows nt 6.0': 'Windows Vista',  'Windows nt 6.1': 'Windows 7',  
              'Windows nt 6.2': 'Windows 8.0',  'Windows nt 6.3': 'Windows 8.1',  
              'X11': 'Linux'}
    # Define user defined functions
    os_general = udf(lambda x: mapper_general[re.findall(ex, x)[0].split(';')[0].capitalize()])
    os_specific = udf(lambda x: mapper_specific[re.findall(ex, x)[0].split(';')[0].capitalize()])
    df = df.withColumn("os_general", os_general(df.userAgent))
    df = df.withColumn("os_specific", os_specific(df.userAgent))
    return df
    
def prepare_df_for_ml(df, saved_as='saved_user_dataset'):

    def create_dummy_df (col, dictionary):
        '''
        Create a dataframe to map a variable
        col: the column name
        dictionary: the mapping of from->to numeric values
        return a dataframe of 2 columns
        '''
        # To map M and F to numeric values, we first should map to string numbers (to avoid spark error)
        col_df = df.select('userId', col).dropDuplicates().replace(dictionary, subset=col)
        # Then convert the result to numeric value
        col_df = col_df.select('userId', col_df[col].cast('int'))
        # Check
        print(col_df.printSchema(), col_df.show(3))
        return col_df
        
    # Create gender column
    gender_df = create_dummy_df('gender', {'M':'1', 'F':'0'})
    
    # create OS columns
    os_titles =  df.select('os_specific').distinct().rdd.flatMap(lambda x: x).collect()
    os_expr = [sF.when(sF.col('os_specific') == osdt, 1).otherwise(0).alias("OS_" + osdt) for osdt in os_titles]
    os_df = df.select('userId', *os_expr)
    
    # Create payment level column
    level_df = create_dummy_df('level', {'paid':'1', 'free':'0'})
    
    # crate song length columns
    song_length = df.filter(df.page=='NextSong').select('userId', 'sessionId', 'length')
    song_length = song_length.withColumn('hours', (song_length.length / 3600))
    song_length = song_length.groupBy('userId', 'sessionId').sum('hours')
    song_length = song_length.groupBy('userId').agg(
                            sF.avg('sum(hours)').alias('mean_hours'), 
                            sF.stddev('sum(hours)').alias('stdev_hours')).na.fill(0)
    
    # crate page visits frequency columns
    # ...................................
    # The distribution of pages per user (FILLING NAN with 0)
    user_page_distribution = df.groupby('userId').pivot('page').count().na.fill(0) #.toPandas().head(30)

    # Drop Cancel    Cancellation Confirmation columns
    user_page_distribution = user_page_distribution.drop(*['Cancel','Cancellation Confirmation'])

    # the columns to be summed
    pages_cols = user_page_distribution.columns[1:]

    # Add a total column
    new_df = user_page_distribution.withColumn('total', sum(user_page_distribution[col] for col in pages_cols))

    # Apply normalization per column
    for col in pages_cols:
        new_df = new_df.withColumn(f'norm_{col}', new_df[col] / new_df['total'] * 100.)
        
    # Remove the total column    
    new_df = new_df.drop('total')

    # Remove the original columns
    new_df = new_df.drop(*pages_cols)

    # Rename the normalized columns back
    oldColumns = new_df.columns
    newColumns = ['userId'] + pages_cols
    user_page_distribution = reduce(lambda new_df, idx: new_df.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), new_df)

    # Freeup memory
    new_df=None
    
    # Number of songs columns
    # number of changing songs
    song_user_df = df.filter(df.page == 'NextSong').groupBy(
                            'userId', 'sessionId').count()
    song_user_df = song_user_df.groupBy('userId').agg(
                            sF.avg('count').alias('mean_songs'), 
                            sF.stddev('count').alias('stdev_songs'))
    song_user_df = song_user_df.na.fill(0)
    
    # Number of artists the user fans
    artists_user_fans = df.select('userId', 'artist').dropDuplicates().groupBy('userId').count().withColumnRenamed("count", "num_aritst")
    
    # Session Duration
    session_end = df.groupBy('userId', 'sessionId').max('ts').withColumnRenamed(
                                                            'max(ts)', 'end')
    session_start = df.groupBy('userId', 'sessionId').min('ts').withColumnRenamed(
                                                                'min(ts)', 'start')
    session_df = session_start.join(session_end,['userId', 'sessionId'])
    ticks_per_hours = 1000 * 60 * 60
    session_df = session_df.select('userId', 'sessionId', ((
        session_df.end-session_df.start)/ticks_per_hours).alias('session_hours'))
    session_user_df = session_df.groupBy('userId').agg(
                        sF.avg('session_hours').alias('mean_session_h'), 
                        sF.stddev('session_hours').alias('stdev_session_h'))
    session_user_df = session_user_df.na.fill(0)
    
    # 2-H Sessions count per user
    num_sessions_user_df = df.select('userId', 'sessionId').dropDuplicates().groupby('userId').count().withColumnRenamed('count', 'num_sessions')
    
    # 2-I The user's subscription age
    def days_since_subscription(df, col_name='days_on'):
        # timestamp of users registration
        reg_ts = df.select('userId', 'registration').dropDuplicates().withColumnRenamed('registration', 'start')
        # reg_ts.show(5)
        # The maximum timestamp found for the user
        end_ts = df.groupBy('userId').max('ts').withColumnRenamed('max(ts)', 'end')
        # end_ts.show(5)
        # The difference
        reg_df = reg_ts.join(end_ts,'userId')
        ticks_per_day = 1000 * 60 * 60 * 24 # as the timestamp is in ticks (0.001 seconds)
        # Merge in one df
        reg_df = reg_df.select('userId', ((reg_df.end-reg_df.start)/ticks_per_day).alias(col_name))
        # reg_ts.join(user_max_ts, user_reg_ts.userId == user_max_ts.userId).select(user_reg_ts["userId"], ((user_max_ts["max(ts)"]-user_reg_ts["registration"])/(1000*60*60*24)).alias("regDay"))
        return reg_df
    # apply the function
    reg_df = days_since_subscription(df, col_name='days_total_subscription')
    
    # the free or paid songs percent per user
    paid_free_df = df.filter(df.page=='NextSong').groupBy('userId').pivot('level').count()
    paid_free_df = paid_free_df.na.fill(0)
    active_cols = paid_free_df.columns[1:]
    paid_free_df = paid_free_df.withColumn('total', 
                                           sum(paid_free_df[col] for col in active_cols))
    for col in active_cols:
        paid_free_df = paid_free_df.withColumn(f'{col}_percent', 
                                               paid_free_df[col] / paid_free_df.total * 100)
    active_cols.append('total')    
    paid_free_df = paid_free_df.drop(*active_cols)
    # It is enough to keep only the paid_percent or the free_percent
    paid_free_df = paid_free_df.drop('free_percent')
    
    # Collect all
    user_features = [gender_df, os_df, paid_free_df, song_length, 
                   user_page_distribution, song_user_df, artists_user_fans, 
                   session_user_df, num_sessions_user_df, reg_df]
    user_features_names = ['gender_df', 'os_df', 'paid_free_df', 'song_length', 
                   'user_page_distribution', 'song_user_df', 'artists_user_fans', 
                   'session_user_df', 'num_sessions_user_df', 'reg_df']
    # Initialize the final_df
    final_df = chorn_users

    def join_features(base, new):
        df_to_join = new#.withColumnRenamed('userId', 't_userId')
        base = base.join(df_to_join, 'userId', how='inner')#.drop('t_userId')#.show(10)
        return base.dropDuplicates()

    for i, feature in enumerate(user_features):
        print(f'Preparing features of the {user_features_names[i]} dataframe', end='; ')
        final_df = join_features(final_df, feature)
        print (f"the new frame's dimensions is: {final_df.count()} * {len(final_df.columns)}")
    final_df = final_df.orderBy('userId', ascending=True)
    print('*** ALL DONE ***')
    
    # Saving the dataframe for future access in another session
    final_df.write.save(f'{saved_as}.CSV', format='csv', header=True)
    print(f'The final dataset was saved as {saved_as}.CSV')

def load_clean_transfer(data_source, save_as='saved_user_dataset'):
    # Read
    df = load_data(data_source)
    # Clean
    df = clean_dataframe(df)
    # Transfer
    prepare_df_for_ml(df, saved_as=save_as)

def load_ml_dataset(saved_as='saved_user_dataset.CSV'):
    final_df = spark.read.csv(saved_as, header = True)
    # Change the column names to strings without spaces
    for col in final_df.columns:
        final_df = final_df.withColumnRenamed(col, col.replace(' ', '_').replace('.', ''))
    # Convert all to numbers as the schema shows strings
    # the first column (userId) would be integer
    final_df = final_df.withColumn('userId', final_df.userId.cast(sT.IntegerType()))
    # All columns from Churn to OS_* should be integer types
    for col in final_df.columns[1:12]:
        final_df = final_df.withColumn(col, final_df[col].cast(sT.IntegerType()))
    # All other columns should be float
    for col in final_df.columns[12:]:
        final_df = final_df.withColumn(col, final_df[col].cast(sT.FloatType()))

    # Remove  nulls from anywhere replacing them by zeros
    final_df = final_df.na.fill(0)        
    return final_df
    
def get_train_test_features(final_df):
    # Get features labels for plotting purposes
    features_labels = final_df.columns[2:]
    # Define the vector assembler for all input columns
    features_vector = VectorAssembler(inputCols=final_df.columns[2:], outputCol='features')
    # Apply the vectorization on the dataset
    input_data = features_vector.transform(final_df)
    features_scaler = StandardScaler(withMean=True, withStd=True, inputCol='features', outputCol='scaled_features')
    features_scaler_fit = features_scaler.fit(input_data)
    scaled_inputs = features_scaler_fit.transform(input_data)
    # Select the output and input features
    ml_data = scaled_inputs.select(scaled_inputs.churn.alias('label'), scaled_inputs.scaled_features.alias('features'))
    # Defining training and testing samples
    train, test = ml_data.randomSplit([0.80, 0.20], seed=179)
    return train, test, features_labels
    
# Defining important functions for outputs and alalysis
# -----------------------------------------------------
def format_duration (t_dif):
    t_s = t_dif.seconds
    duration = {}
    duration['h'], rem = divmod(t_s, 3600) 
    duration['m'], duration['s'] = divmod(rem, 60)
    stamp = ''
    if duration['h']>0:
        stamp += f"{duration['h']} hour(s), " 
    if duration['m']>0:
        stamp += f"{duration['m']} minute(s) and "
    # seconds and fraction of seconds
    frac = int(t_dif.microseconds/10000)/100
    stamp += f"{duration['s'] + frac} second(s)"
    # print(f"{duration['h']}h:{duration['m']}m:{duration['s']}s")
    return stamp
    
def model_fitting(data, model_type, param_grid, save_as, num_folds=3, random_seed=179):
    '''
    
    
    '''
    model_evaluator = CrossValidator(estimator=model_type, estimatorParamMaps=param_grid,
                                      evaluator=MulticlassClassificationEvaluator(),
                                      numFolds=num_folds, seed=random_seed)
    t_start = pd.tslib.Timestamp.now()
    print ('Fitting in progress...', end=' ')
    fitted_model = model_evaluator.fit(data)
    t_dif = pd.tslib.Timestamp.now() - t_start
    print (f'Done in {format_duration(t_dif)}')
    t_start = pd.tslib.Timestamp.now()
    print (f'\nSaving the model as {save_as}...' , end=' ')
    try:
        fitted_model.save(save_as)
    except:
        # Overwrite if exists
        fitted_model.write().overwrite().save(save_as)
        print ('*Overwritten* ', end='')
    t_dif = pd.tslib.Timestamp.now() - t_start
    print (f'Done in {format_duration(t_dif)}')
    return fitted_model    
        
def get_formated_metrics(selected_model, test_data):
    '''
    Prints a compacted dataframe with all the model's metrics
    selected_model: The fitted model
    test_data: the test data portion
    '''
    def get_model_metrics(selected_model, model_type = 'train'):
        '''
        Get the metrics of a model
        selected_model:  the fitted model
        model_type: either 'train' (default) or 'test'
        '''
        if model_type == 'train':
            metrics = selected_model.bestModel.summary
        else: 
            metrics = selected_model
        acc = metrics.accuracy, 
        general = np.array((metrics.weightedFMeasure(),
                   metrics.weightedPrecision, metrics.weightedRecall,
                   metrics.weightedTruePositiveRate, metrics.weightedFalsePositiveRate))
        general = general.reshape(1, general.shape[0])
        labels = ['General'] + [f'Churn={x}' for x in metrics.labels]
        labeled = np.array((metrics.fMeasureByLabel(),
                          metrics.precisionByLabel, metrics.recallByLabel,
                          metrics.truePositiveRateByLabel, metrics.falsePositiveRateByLabel))
        conc_results = np.concatenate((general.T, labeled), axis=1)
        metrics_names = ['F-Measure', 'Precision', 'Recall', 'True_+ve_Rate', 'False_+ve_Rate']
        df_res = pd.DataFrame(conc_results, columns=labels, index=metrics_names)
        return acc[0], df_res
    
    # Apply for training data
    acc_train, train_res = get_model_metrics(selected_model)
    # Get the results of the test data
    model_test = selected_model.bestModel.evaluate(test_data)
    # Apply on test data
    acc_test, test_res = get_model_metrics(model_test, model_type='test')
    
    # Concatenate to a pretty dataframe
    pretty_frame = pd.concat([train_res, test_res], axis=1, keys=[
                    f'Training (Accuracy = {acc_train*100:4.2f}%)',
                    f'Testing (Accuracy = {acc_test*100:4.2f}%)'])
    return pretty_frame
    
def draw_features_contribution(fitted_model, x_labels, scale_to='full_range'):
    '''
    Draws a bar chart of features vs churn %
    fitted_model: the fitted model
    scale_to: the values will be scated to:
           'full_range' where the full absolute values are summed to 100.
           'maximum_range' where the maximum absolute extremes are scalled to 100.
           'none' the values are shown as is.
    '''
    cmx = fitted_model.bestModel.coefficientMatrix
    cmv = cmx.values
    # cmv.shape, len(final_df.columns[2:])
    
    # Define positive and negative values
    positives_v = np.array([x if x>=0 else 0 for x in cmv])
    negatives_v = np.array([x if x<=0 else 0 for x in cmv])    
    
    # Drawing by scalling the maximum range to  100
    if scale_to == 'full_range':
        rang = positives_v.sum()+ abs(negatives_v).sum()
    elif scale_to == 'maximum_range':
        rang = positives_v.max()+ abs(negatives_v).max()
    else:
        rang = 1.
        
    positives_v /= rang
    negatives_v /= rang
    positives_v *= 100.
    negatives_v *= 100.
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(x_labels, positives_v,color='r')
    ax.bar(x_labels, negatives_v, color='g')
    ax.set_xlabel('Features')
    ax.set_ylabel('The user is most likely to churn (%)')
    ax.set_title('Contribution of each feature to the churn decission')
    ax.set_xticklabels(labels = final_df.columns[2:], rotation='vertical');
    plt.show()
    
def get_classifier_metrics(trained_model, train_data, test_data):
    '''
    
    '''
    def get_specific_metrics(trained_model, data):
        '''
        
        '''
        res2 = trained_model.transform(data).select('label', 'prediction')
        TruePos = res2.filter((res2.prediction==1)& (res2.label == res2.prediction) ).count()
        TrueNeg = res2.filter((res2.prediction==0)& (res2.label == res2.prediction) ).count()
        FalsPos = res2.filter((res2.prediction==1)& (res2.label != res2.prediction) ).count()
        FalsNeg = res2.filter((res2.prediction==0)& (res2.label != res2.prediction) ).count()
        accuracy = res2.filter(res2.label == res2.prediction).count()/res2.count()
        precision = TruePos/(TruePos+FalsPos)
        recall = TruePos/(TruePos+FalsNeg)
        f1score = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1score
    train_metrics = get_specific_metrics(trained_model, train_data)
    test_metrics = get_specific_metrics(trained_model, test_data)
    labels =['Train', 'Test']
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F-Score']
    metrics_data =np.array((train_metrics, test_metrics))
    return pd.DataFrame(data=metrics_data.T, columns=labels, index=metrics_names)

def draw_features_importance(fitted_model, x_labels, threshold=0.1):
    '''
    Draws a pie chart of features
    fitted_model: the fitted model
    x_labels: the labels of the features.
    threshold: the minimum value (%) to consider, 
               if the value is less than that, 
               it will be neglected (default =0)
    '''
    importance = list(fitted_model.bestModel.featureImportances.toArray())
    # Get the threshold value
    thres_v = threshold / 100 
    # get the included and neglected values
    active_values = [x for x in importance if x >= thres_v]
    neglected = [x for x in importance if x < thres_v]
    non_zero_neglected = [x for x in neglected if x > 0]
    # print(importance, '\n', x_labels, '\n', thres_v, '\n', 
    #       active_values, '\n', neglected, '\n', non_zero_neglected)
    # get the accepted indexes
    active_idx = [importance.index(x) for x in active_values]
    # the accepted lables + minor features
    active_labels = [x_labels[x] for x in active_idx]
    minor_v = sum(neglected)
    # print(active_idx, '\n', active_labels, '\n', minor_v)
    # If there is any minor features
    if minor_v>0:
        active_values.append(minor_v)
        active_labels.append(f'MINOR, ({len(non_zero_neglected)}feats. each<{threshold}%)')
    # print(active_labels, '\n', active_values)
    
    # sorting
    active_labels =[x for _, x in sorted(zip(active_values, active_labels))]
    active_values = sorted(active_values)
    
    # Draw

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.pie(active_values[::-1], labels=active_labels[::-1] , 
           autopct='%1.1f%%', shadow=True, 
           startangle=90 )
    ax.set_title('Importance of each feature to the churn decission')
    ax.axis('equal');
    plt.show();
    
def apply_model(train, test, features_labels, 
                model_name='LogisticRegression', 
                load_from_existing = None,
                save_as = None,
                parameter_grid=None):
    '''
    This function can do two things:
      1. Apply a ML model to the train data and save the model
      2. Load an existing model
    Then it applies the model to the test data, and 
        1. prints a table of the model metrics
        2. draw a bar chart or a pie chart with features importance
    Inputs:
        train: The training dataset (spark format)
        test: The testing dataset (spark format)
        features_labels: the features names(list of strings)
        model_name='LogisticRegression': The name of the model, it can be:
            1. LogisticRegression or LR
            2. DecisionTreeClassifier or DTC
            3. GradientBoostedTrees or GBT
            4. RandomForestClassifier or RFC
            5. MultilayerPerceptronClassifier or MPC {This model will only display a table, no figure drawn}
            all the names are NOT case sensitive
        load_from_existing = None: If not none, then the model will be loaded from the given path
        save_as = None: If not None, the model will be saved to the given name, 
                    otherwise it will be save as '{model_name}.model'
        parameter_grid=None: if None, a default param_grid will be used for each model, 
                    otherwise, the given grid will be used.
    '''
    if save_as is None:
        save_as = f'{model_name}.model'
        
    if (model_name.lower() == 'logisticregression')|(model_name.lower() == 'lr'):
        # The Logistic Regression model
        if load_from_existing is None:
            model = LogisticRegression()
            param_grid = ParamGridBuilder() \
                .addGrid(model.regParam,[0.01, 0.1]) \
                .addGrid(model.elasticNetParam,[0.0, 0.5]) \
                .addGrid(model.aggregationDepth,[2, 5]) \
                .build()
            if parameter_grid is not None:
                param_grid = parameter_grid

            m = model_fitting(train, model, param_grid, save_as)
        else:
            m = mlm.load(load_from_existing)
        # Model metrics
        display(get_formated_metrics(m, test))
        # Features effect
        draw_features_contribution(m, x_labels=features_labels)
        
    elif (model_name.lower() == 'decisiontreeclassifier')|(model_name.lower() == 'dtc'):
        # The Decision Tree Classifier model
        if load_from_existing is None:
            model = DecisionTreeClassifier()
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxDepth,[3, 5, 10]) \
                .addGrid(model.impurity,['entropy', 'gini']) \
                .build()
            if parameter_grid is not None:
                param_grid = parameter_grid
            m = model_fitting(train, model, param_grid, save_as)
        else:
            m = mlm.load(load_from_existing)
        display(get_classifier_metrics(m, train, test))
        draw_features_importance(m, features_labels, threshold=3)
    elif (model_name.lower() == 'gradientboostedtrees')|(model_name.lower() == 'gbt'):    
        # The Gradient-Boosted Trees (GBTs) model
        if load_from_existing is None:
            model = GBTClassifier()
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxDepth,[3, 5, 10]) \
                .addGrid(model.maxBins ,[10, 5]) \
                .addGrid(model.maxIter ,[20, 5]) \
                .build()
            if parameter_grid is not None:
                param_grid = parameter_grid
            m = model_fitting(train, model, param_grid, save_as)
        else:
            m = mlm.load(load_from_existing)
        display(get_classifier_metrics(m, train, test))
        draw_features_importance(m, features_labels, threshold=3)
    elif (model_name.lower() == 'randomforestclassifier')|(model_name.lower() == 'rfc'):    
        # The Random Forest model
        if load_from_existing is None:
            model = RandomForestClassifier()
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxDepth,[5, 10]) \
                .addGrid(model.impurity,['entropy', 'gini']) \
                .addGrid(model.maxBins ,[10, 5]) \
                .addGrid(model.numTrees ,[20, 10]) \
                .addGrid(model.featureSubsetStrategy ,['sqrt', 'onethird']) \
                .build()
            if parameter_grid is not None:
                param_grid = parameter_grid
            m = model_fitting(train, model, param_grid, save_as)
        else:
            m = mlm.load(load_from_existing)
        display(get_classifier_metrics(m, train, test))
        draw_features_importance(m, features_labels, threshold=3)        
    elif (model_name.lower() == 'multilayerperceptroncclassifier')|(model_name.lower() == 'mpc'):    
        # The Multilayer Perceptron Classifier model
        if load_from_existing is None:
            model = MultilayerPerceptronClassifier()
            param_grid = ParamGridBuilder() \
                .addGrid(model.blockSize,[64, 128]) \
                .addGrid(model.maxIter,[10, 20]) \
                .addGrid(model.stepSize ,[0.03, 0.01]) \
                .addGrid(model.solver ,['l-bfgs', 'gd']) \
                .addGrid(model.layers, [[37, 12, 2], [37, 5, 2]]) \
                .build()
            if parameter_grid is not None:
                param_grid = parameter_grid
            m = model_fitting(train, model, param_grid, save_as)
        else:
            m = mlm.load(load_from_existing)
        display(get_classifier_metrics(m, train, test))
        # draw_features_importance(m, features_labels, threshold=3)            