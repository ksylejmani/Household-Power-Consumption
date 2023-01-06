# Import relevant modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder


# Adjust granularity of reporting
pd.options.display.max_rows=10
pd.options.display.float_format="{:.1f}".format

class Solution:
    def __init__(self,file_name, train_size,label_name) -> None:
        """Load the dataset, split it into training and test set, shuffle the training set"""
        self.dataset_df=self.get_data_from_file(file_name)
        self.train_df, self.test_df=self.split_data_set(train_size)
        self.train_df_norm, self.test_df_norm=self.normalize_values()
        self.feature_layer=self.create_feature_layer(label_name)

    def get_data_from_file(self,file_name):
        """Read dataset from text file"""
        with open(file_name+".txt",encoding="utf8") as open_file_name:
            dataset_df=pd.read_csv(open_file_name,delimiter=';',parse_dates=['Date'])
        
        # Convert object types to respectiv types
        dataset_df['Time']=pd.to_datetime(dataset_df['Time'],format='%H:%M:%S')
        dataset_df['Global_active_power']=pd.to_numeric(dataset_df['Global_active_power'],errors='coerce')
        dataset_df['Global_reactive_power']=pd.to_numeric(dataset_df['Global_reactive_power'],errors='coerce')
        dataset_df['Voltage']=pd.to_numeric(dataset_df['Voltage'],errors='coerce')
        dataset_df['Global_intensity']=pd.to_numeric(dataset_df['Global_intensity'],errors='coerce')
        dataset_df['Sub_metering_1']=pd.to_numeric(dataset_df['Sub_metering_1'],errors='coerce')
        dataset_df['Sub_metering_2']=pd.to_numeric(dataset_df['Sub_metering_2'],errors='coerce')
        dataset_df['Sub_metering_3']=pd.to_numeric(dataset_df['Sub_metering_3'],errors='coerce')

        # Drop NaN values
        dataset_df=dataset_df.dropna(axis=0)
        return dataset_df
    
    def split_data_set(self,train_size):
        """Split data set into training and test set"""
        # Shuffle dataset
        self.dataset_df=self.dataset_df.reindex(np.random.permutation(self.dataset_df.index))

        # Calculate number of rows for train set
        num_rows_train=int(len(self.dataset_df.index)*train_size)
        
        # Split data set
        train_df=self.dataset_df.iloc[0:num_rows_train]
        test_df=self.dataset_df.iloc[num_rows_train:]
        
        # Shuffle train set
        train_df=train_df.reindex(np.random.permutation(train_df.index))
        
        return train_df, test_df
    
    def normalize_values(self):
        """Convert the raw values to their Z-scores for applicable features"""
        applicable_features=['Global_active_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
        
        # Add features that do not need to be normalized
        train_df_norm=pd.DataFrame(self.train_df['Global_reactive_power'])
        test_df_norm=pd.DataFrame(self.test_df['Global_reactive_power'])
        
        # Normalize train set
        train_df_mean=self.train_df[applicable_features].mean()
        train_df_std=self.train_df[applicable_features].std()
        train_df_norm[applicable_features]=(self.train_df[applicable_features]-train_df_mean)/train_df_std

        # Normalize test set
        test_df_mean=self.test_df[applicable_features].mean()
        test_df_std=self.test_df[applicable_features].std()
        test_df_norm[applicable_features]=(self.test_df[applicable_features]-test_df_mean)/test_df_std

        # Add one hot encoded columns for hour (i.e. 24 columns for each hour of the day)
        train_one_hot_columns, test_one_hot_columns=self.create_time_feature_column()
        train_df_norm=pd.concat([train_df_norm,train_one_hot_columns],axis=1)
        test_df_norm=pd.concat([test_df_norm,test_one_hot_columns],axis=1)
        
        # Add one hot encoded columns for each week of the year (i.e. 53 columns for each week of the year)
        train_one_hot_columns, test_one_hot_columns=self.create_week_feature_column()
        train_df_norm=pd.concat([train_df_norm,train_one_hot_columns],axis=1)
        test_df_norm=pd.concat([test_df_norm,test_one_hot_columns],axis=1)

        print(train_df_norm.describe())
        print(test_df_norm.describe())

        return train_df_norm, test_df_norm
    
    
    def create_feature_layer(self,label_name):
        """Create the feature layer using tf.feature_column"""
        feature_columns=[]
        for c in self.train_df_norm.columns:
            if c !=label_name:
                current_feature=tf.feature_column.numeric_column(c)
                feature_columns.append(current_feature)
        feature_layer=layers.DenseFeatures(feature_columns)
        return feature_layer

    def create_linear_model(self,learning_rate,l2_regularization):
        """Create and compile a simple linear regression model."""
        model=tf.keras.models.Sequential()
        model.add(self.feature_layer)
        model.add(tf.keras.layers.Dense(units=1,input_shape=(1,),kernel_regularizer=tf.keras.regularizers.l2(l=l2_regularization)),)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                    loss="mean_squared_error",
                    metrics=[tf.keras.metrics.MeanSquaredError()])
        return model

    def create_time_feature_column(self):
        """Create a feature column by backetizing the 'Time' feature in 24 backets"""
        feature_names={i:'Hour_'+"{:1d}".format(i) for i in range(24)}
        one_hot_encoder=OneHotEncoder(handle_unknown='ignore',sparse=False)
        
        # Create feature column for train set
        train_df_hour=pd.DataFrame()
        train_df_hour['Hour']=(pd.to_datetime(self.train_df['Time']).dt.hour)
        train_one_hot_columns_df=pd.DataFrame(one_hot_encoder.fit_transform(train_df_hour[['Hour']]))
        train_one_hot_columns_df.index=train_df_hour.index
        train_one_hot_columns_df.rename(columns=feature_names,inplace=True)
        
        # Create feature column for test set
        test_df_hour=pd.DataFrame()
        test_df_hour['Hour']=(pd.to_datetime(self.test_df['Time']).dt.hour)
        test_one_hot_columns_df=pd.DataFrame(one_hot_encoder.fit_transform(test_df_hour[['Hour']]))
        test_one_hot_columns_df.index=test_df_hour.index
        test_one_hot_columns_df.rename(columns=feature_names,inplace=True)

        return train_one_hot_columns_df, test_one_hot_columns_df

    def create_week_feature_column(self):
        """Create a feature column by backetizing the 'Week' feature in 52 backets"""
        feature_names={i:'Week_'+"{:1d}".format(i) for i in range(53)}
        one_hot_encoder=OneHotEncoder(handle_unknown='ignore',sparse=False)
        
        # Create feature column for train set
        train_df_week=pd.DataFrame()
        train_df_week['Week']=(pd.to_datetime(self.train_df['Date']).dt.week)
        train_one_hot_columns_df=pd.DataFrame(one_hot_encoder.fit_transform(train_df_week[['Week']]))
        train_one_hot_columns_df.index=train_df_week.index
        train_one_hot_columns_df.rename(columns=feature_names,inplace=True)
        
        # Create feature column for test set
        test_df_hour=pd.DataFrame()
        test_df_hour['Week']=(pd.to_datetime(self.test_df['Date']).dt.week)
        test_one_hot_columns_df=pd.DataFrame(one_hot_encoder.fit_transform(test_df_hour[['Week']]))
        test_one_hot_columns_df.index=test_df_hour.index
        test_one_hot_columns_df.rename(columns=feature_names,inplace=True)
        return train_one_hot_columns_df, test_one_hot_columns_df
    
    def train_linear_model(self,model,epochs,batch_size,label_name):
        """Feed a dataset into the model in order to train it."""
        # Split data set into features and label
        train_features={name:np.array(value) for name, value in self.train_df_norm.items()}
        train_label=np.array(train_features.pop(label_name))
        history=model.fit(x=train_features,y=train_label,batch_size=batch_size,epochs=epochs,shuffle=True)
        epochs=history.epoch
        hist=pd.DataFrame(history.history)
        mse=hist["mean_squared_error"]
        return epochs, mse
    
    def evaluate_linear_model(self,model,label_name,batch_size):
        """Evaluate the model using test set and batch_size"""
        test_features={name: np.array(value) for name, value in self.test_df_norm.items()}
        test_label=np.array(test_features.pop(label_name)) #Isolate the label
        model_evaluation_metrics=model.evaluate(x=test_features,y=test_label,batch_size=batch_size)
        return model_evaluation_metrics

    def call_linear_model(self,learning_rate,epoch,batch_size,label_name,l2_regularization):
        """Call model using the set of hyperparameters"""
        model=self.create_linear_model(learning_rate,l2_regularization)
        epochs,mse=self.train_linear_model(model,epoch,batch_size,label_name)
        linear_evaluation=self.evaluate_linear_model(model,label_name,batch_size)
        return [epochs, mse, mse.min(), mse.max(),"Linear model",linear_evaluation] 
    
    def create_deep_neural_net(self,learning_rate,dropout_rate,l2_regularization):
        """Create a deep neural net"""
        model=tf.keras.models.Sequential()
        print(self.feature_layer)
        model.add(self.feature_layer)
        
        # Define the first hidden layer with 20 nodes
        model.add(tf.keras.layers.Dense(units=20, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l=l2_regularization),name="Hidden1"))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))

        # Define the second hidden layer with 8 nodes
        model.add(tf.keras.layers.Dense(units=8,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l=l2_regularization),name="Hidden2"))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        
        # Define the third hidden layer with 12 nodes
        model.add(tf.keras.layers.Dense(units=12,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l=l2_regularization),name="Hidden3"))

        # Define the output layer
        model.add(tf.keras.layers.Dense(units=1,name="Output"))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])
        return model

    def train_deep_neural_net(self,model,epochs,label_name,batch_size=None):
        """Train the deap neural net by feedign it data"""
        # Split the dataset into features and label
        train_featues={name:np.array(value) for name, value in self.train_df_norm.items()}
        train_label=np.array(train_featues.pop(label_name))
        history=model.fit(x=train_featues,y=train_label,batch_size=batch_size,epochs=epochs, shuffle=True)
        epochs=history.epoch
        hist=pd.DataFrame(history.history)
        mse=hist["mean_squared_error"]
        return epochs,mse  

    def evaluate_deep_neural_model(self,model,label_name,batch_size):
        """Evaluate deep neural model using test set"""
        test_features={name:np.array(value) for name, value in self.test_df_norm.items()}
        test_label=np.array(test_features.pop(label_name))
        model_evaluation_metrics=model.evaluate(x=test_features,y=test_label,batch_size=batch_size)
        return model_evaluation_metrics

    def call_deep_neural_net_model(self,learning_rate,epochs,batch_size,label_name,dropout_rate,l2_regularization):
        """Call deep neural net model by using hyperparameters"""
        model=self.create_deep_neural_net(learning_rate,dropout_rate,l2_regularization)
        epochs, mse=self.train_deep_neural_net(model,epochs,label_name,batch_size)
        deep_net_evaluation=self.evaluate_deep_neural_model(model,label_name,batch_size)
        return [epochs, mse, mse.min(),mse.max(), "Deep net model",deep_net_evaluation]

    def plot_loss_curve(self,result_list,epochs):
        """Plot a curve of loss vs. epoch."""
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        for c in result_list:
            rgb = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(c[0],c[1],label=c[4]+' training loss',color=rgb)
            plt.hlines(c[5][1],xmin=0, xmax=epochs,linestyles='dashed',colors=[rgb],label=c[4]+' evaluation loss')
        plt.legend()
        plt.ylim([min(row[2] for row in result_list)*0.01,max(row[3] for row in result_list)*1.05])
        plt.show()

if __name__=="__main__":
    """Set model parameters and instance name and create an instance of the model"""
    
    file_name="household_power_consumption"

    # Set hyperparameters
    learning_rate=0.01
    batch_size=5000
    train_size=0.8
    epochs=50
    dropout_rate=0.05
    l2_regularization=0.001
    label_name="Global_reactive_power"

    # Create an instance of the model and call it
    s=Solution(file_name,train_size,label_name)
    result_list=[]
    result_list.append(s.call_linear_model(learning_rate,epochs,batch_size,label_name,l2_regularization))
    result_list.append(s.call_deep_neural_net_model(learning_rate,epochs,batch_size,label_name,dropout_rate,l2_regularization))
    s.plot_loss_curve(result_list,epochs)