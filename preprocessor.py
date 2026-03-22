import pandas as pd
import numpy as np
import chardet
from sklearn.impute import KNNImputer


class Model_Preprocessor():
    def __init__(self, model_name):
        self.model = model_name

    def read_dataset(self,filename: str, col_names: dict):

        """
        Function for reading a CSV file into a dataframe and performing high-level transformations.

        Parameters:

        filename: str -> Name of the CSV file to be read.
        col_names: dict -> A dictionary of column names with the desired data type for transformation.
        
        """
        with open(filename, 'rb') as f:
            res = chardet.detect(f.read())['encoding']
            
        df = pd.read_csv(filename, encoding = res)
        fil_df = df[col_names.keys()].copy()
        fil_df = fil_df.astype(col_names)
        fil_df.columns = [col.lower() for col in fil_df.columns]
        
        return fil_df

    def date_range(self, start_date: str, end_date: str):

        """
        Creates a calendar table to be used to map all the values.
        
        Parameters:
        
        start_date: str -> Starting date as a String.
        end_date: str -> Ending date as a String.
        
        OUTPUT:
        
        Returns a Pandas Dataframe with a single column named ds with the datetime elements evenly
        recorded at 30-minute intervals.
        
        """
        
        start_date = start_date
        end_date = end_date
        
        table = pd.DataFrame(pd.date_range(start = start_date, end = end_date, freq = "30min"),
                             columns = ['ds']
                            )
        
        return table.sort_values(by = 'ds')
        

    def add_datetime_features(self, df: pd.DataFrame, date_col: str):

        """
        Creates a datetime table on a 30 mins iterval based on the dataframe provided

        Parameters:

        df: pd.DataFrame -> Name of the dataframe on which to create the table.
        date_col: str -> The dates column to use for calculating range.
        interval (Default): str -> Creates a dates table on a 30 mins interval.

        OUTPUT:

        Returns a Pandas dataframe with the column named as ds.
        
        """

        seasons = {12: "Summer", 1: "Summer", 2: "Summer", 3: "Autumn", 4: "Autumn", 5: "Autumn", 6: "Winter", 7: "Winter", 8: "Winter",
                  9: "Spring", 10: "Spring", 11: "Spring"}

        time_of_day = {0: "Late Evening", 1: "Night", 2: "Night", 3: "Night", 4: "Night", 5: "Early Morning", 6: "Early Morning",
                      7: "Early Morning", 8: "Early Morning", 9: "Morning", 10: "Morning", 11: "Morning", 12: "Morning",
                      13: "Afternoon", 14: "Afternoon", 15: "Afternoon", 16: "Afternoon", 17: "Evening", 18: "Evening",
                      19: "Evening", 20: "Evening", 21: "Late Evening", 22: "Late Evening", 23: "Late Evening"}
        
        df['hour'] = df[date_col].dt.hour
        df['month'] = df[date_col].dt.month
        df['season'] = df['month'].map(seasons).astype('category')
        df['timeofday'] = df['hour'].map(time_of_day).astype('category')

        df.drop(['hour', 'month'], axis = 1, inplace = True)
        
        return df


    def remove_duplicates(self, df: pd.DataFrame, col_name: str):

        """
        Checks and removes duplicate values from a dataframe.

        Parameters:
        df: pd.DataFrame -> Dataframe that needs to be cleaned.
        col_name: str -> The cloumn that is required for identifying duplicate entries.
        
        """
        
        df['min'] = df['datetime'].dt.minute
        fil_temp = df[df['min'].isin([0,30])].copy()
        fil_temp.drop_duplicates(subset = [col_name], keep = 'last', inplace = True)

        fil_temp.drop('min', axis = 1, inplace = True)
    
        return fil_temp


    def fillin_missing_values(self, df: pd.DataFrame):

        """
        This function uses the KNNImputer to impute missing values in the dataset.

        Parameters:

        df: pd.DataFrame -> The dataframe to operate and fill in Nan's.

        OUTPUT:

        Returns a cleaned dataframe with all the missing values imputed.
        
        """

        # finding the columns in the dataframe with missing values.
        
        col_value_counts = {col: df[col].isnull().sum() for col in df.columns}
        missing_cols = [key for key, value in col_value_counts.items() if value != 0]
    
        # setting up the imputer.
        
        imputer = KNNImputer(n_neighbors = 4)
    
        # filling in the missing values using the imputer.
        
        for col in missing_cols:
            df[f"Imputed_{col}"] = imputer.fit_transform(df[col].values.reshape(-1,1))
        
        df.drop(missing_cols, axis = 1, inplace = True)
        df.rename(columns = {'datetime': 'ds', 'Imputed_temperature': 'temp', 'Imputed_totaldemand' : 'totaldemand'}, inplace = True)
            
        return df

    def resample_data(self, df: pd.DataFrame, response_var_name: str, encode_categorical = 0):

        """
        Function to generate the actual dataframe to be used for model training and testing. 
        Aggregates the original data in hourly frequency.

        Parameters:

        df: pd.DataFrame -> The dataframe to be resampled.
        mapping_df: pd.DataFrame -> Additional dataframe to adding season and daily attributes.

        OUTPUT:

        Resampled (1 hr frequency) Pandas DataFrame with seasonal and daily attributes.
        
        """
        if encode_categorical:
        
            encoded_df = pd.get_dummies(df, dtype = 'int8')  
            agg_df = encoded_df.resample(rule = 'h', on = 'ds').mean()
            agg_df.rename(columns = {response_var_name: 'y'}, inplace = True)
            agg_df = agg_df[['ds', 'season_autumn', 'season_spring', 'season_summer', 'season_winter',
                              'timeofday_afternoon', 'timeofday_early morning', 'timeofday_evening', 
                              'timeofday_late evening', 'timeofday_morning', 'timeofday_night', 
                              'totaldemand', 'y']]
        
            return agg_df.sort_values(by = 'ds')
        
        else:
            
            col_dtypes = {col: df[col].dtypes for col in df.columns}
            num_cols = [col for col in col_dtypes.keys() if col_dtypes.get(col) != 'category']
        
            fil_df = df[num_cols].copy()   
            agg_df = fil_df.resample(rule = 'h', on = 'ds').mean().reset_index()
            agg_df.rename(columns = {response_var_name: 'y'}, inplace = True)

            return agg_df.sort_values(by ="ds")