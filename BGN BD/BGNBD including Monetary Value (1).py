# Databricks notebook source
# MAGIC %md
# MAGIC #Importing packages

# COMMAND ----------

!pip install lifetimes

# COMMAND ----------

import pandas as pd   
import numpy as np                                          #used for data manipulation and analysis
import seaborn as sns                                           #visualization library
import matplotlib.pyplot as plt                                 #visualizations
from matplotlib.pyplot import figure 
from lifetimes import BetaGeoFitter                             #implements the Beta-Geometric/Negative Binomial model for predicting the RFM values. 
from lifetimes import GammaGammaFitter                          #For Predicting monetary value
from lifetimes.utils import summary_data_from_transaction_data    #summarize transaction data into a format suitable for model fitting.
from lifetimes.plotting import plot_frequency_recency_matrix    #visualizations
from lifetimes.plotting import plot_probability_alive_matrix    #visualizations
from lifetimes.plotting import plot_period_transactions         #visualizations
from lifetimes.plotting import plot_history_alive               #visualizations
import warnings

# COMMAND ----------

# MAGIC %md
# MAGIC #Getting Transactional Data

# COMMAND ----------

# %sql
# create or replace table sandbox.ps_trans_data_BDNBG as 
# (select 
#         pt.customer_id,
#         transaction_id,
#         segment,
#         business_day  as transaction_date,
#         round(sum(amount),1) as transaction_value
        
# from gold.pos_transactions pt
# left join gold.material_master mm 
# on pt.product_id  = mm.material_id
# left join analytics.customer_segments cs
# on pt.customer_id = cs.customer_id
# where pt.business_day > '2023-03-01'
# and pt.amount > 0 
# and quantity > 0
# and key = 'rfm'
# and channel = 'pos'
# and cs.country = 'uae'
# and month_year = 202404
# and pt.customer_id is not null
# and transaction_type in ('SALE','SELL_MEDIA')
# -- and customer_id in (select customer_id from sandbox.am_vip_freq_sm)  
# and upper(mm.department_class_name) IN ('SUPER MARKET','FRESH FOOD')
# group by segment,pt.customer_id,transaction_id,business_day)

# COMMAND ----------

# MAGIC %md
# MAGIC #VIP

# COMMAND ----------

# MAGIC %md
# MAGIC ##VIP transactional data

# COMMAND ----------

sdf=spark.sql("""select customer_id,transaction_id,transaction_date, transaction_value from sandbox.ps_trans_data_bdnbg where segment='VIP'""")
df_vip = sdf.toPandas()

# COMMAND ----------

df_vip.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Splitting into train and test

# COMMAND ----------

train_data = df_vip[df_vip['transaction_date'] < '2024-05-01'] 
test_data = df_vip[(df_vip['transaction_date'] >= '2024-05-01') & (df_vip['transaction_date'] <= '2024-05-31')] 

print("Start Train dataset date {}".format(train_data["transaction_date"].min()))
print("End Train dataset date {}".format(train_data["transaction_date"].max()))
print("---------------------------------------------")
print("Start Test dataset date {}".format(test_data["transaction_date"].min()))
print("End Test dataset date {}".format(test_data["transaction_date"].max()))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Summarizing Transactional Data

# COMMAND ----------

#This function generates the recency, frequency, monetary values and tenure for the given train data customer IDs
df_vip_summarized = summary_data_from_transaction_data(train_data, 
                                         'customer_id', 
                                         'transaction_date', 
                                         'transaction_value',
                                         observation_period_end='2024-04-30')
df_vip_summarized.reset_index(level=0, inplace=True)
df_vip_summarized.shape

# COMMAND ----------

df_vip_summarized.head()

# COMMAND ----------

df_vip_summarized['frequency'].describe(percentiles= [0.99])

# COMMAND ----------

df_vip_summarized['recency'].describe(percentiles= [0.99])

# COMMAND ----------

# MAGIC %md
# MAGIC ##Plots

# COMMAND ----------

plt.hist(df_vip_summarized['recency'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Recency')
plt.title('Distribution of Recency-VIP')
plt.show()
plt.savefig('recency.png')

# COMMAND ----------

sns.histplot(df_vip_summarized['frequency'], binwidth=10, color= 'red', edgecolor='black')
plt.xlabel('frequency')
plt.title('Distribution of frequency-VIP')
plt.show()

# COMMAND ----------

plt.hist(df_vip_summarized['monetary_value'], bins=30, edgecolor='k',range=(0, 1000))
plt.title('Spend Distribution-VIP')
plt.xlabel('Monetary Value')
plt.xlim(0, 1000)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Fitting the data to the model

# COMMAND ----------

bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(df_vip_summarized['frequency'], df_vip_summarized['recency'], df_vip_summarized['T'])
bgf.summary

# COMMAND ----------

# MAGIC %md
# MAGIC ##Predicting Visits for the next month

# COMMAND ----------

t = 30 #30 days
df_vip_summarized['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                                      df_vip_summarized['frequency'], 
                                                                                      df_vip_summarized['recency'], 
                                                                                      df_vip_summarized['T'])

# COMMAND ----------

df_vip_summarized['predicted_purchases'] = np.round(df_vip_summarized['predicted_purchases'],0)

# COMMAND ----------

df_vip_summarized[['customer_id','predicted_purchases']].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Predicting Monetary Value(Revenue) for next month

# COMMAND ----------

df_vip_summarized_positive = df_vip_summarized[df_vip_summarized['monetary_value'] > 0]

# COMMAND ----------

ggf = GammaGammaFitter(penalizer_coef = 0.1)
ggf.fit(df_vip_summarized_positive['frequency'],
        df_vip_summarized_positive['monetary_value'])  
print(ggf)

# COMMAND ----------

df_vip_summarized["predicted_amount"] = ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    df_vip_summarized['frequency'],
    df_vip_summarized['recency'],
    df_vip_summarized['T'],
    df_vip_summarized['monetary_value'],
    time= 1,
    discount_rate=0.01 # monthly discount rate
)


# COMMAND ----------

df_vip_summarized.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Plot

# COMMAND ----------

plt.hist(df_vip_summarized['predicted_purchases'], bins=20, color= 'red', edgecolor='black')
plt.xlabel('Number of Visits')
plt.title('Distribution of VIPs Predicted Number of Visits-VIP')
plt.show()

# COMMAND ----------

plt.hist(df_vip_summarized['predicted_amount'], bins=30, edgecolor='k',range=(0, 1000))
plt.title('Predicted amount Distribution-VIP')
plt.xlabel('Spend')
plt.xlim(0, 1000)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Getting Actual Visits

# COMMAND ----------

df_visits = test_data.groupby('customer_id').nunique()['transaction_date'].reset_index()
df_visits.rename(columns={'transaction_date': 'actual_visits'}, inplace=True)

# COMMAND ----------

df_visits.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Getting Actual Spend

# COMMAND ----------

df_amount = (
    test_data.groupby(["customer_id"])[["transaction_value"]]
    .agg("sum")
    .reset_index()
    .rename(columns={"transaction_value": "actual_amount"})
)

# COMMAND ----------

df_amount.head()

# COMMAND ----------

# Merge the two DataFrames of actual visits and actual amount
df_vip_test_summarized = pd.merge(df_visits, df_amount, on='customer_id')

# COMMAND ----------

df_vip_test_summarized.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Merging for predicted and actual visits

# COMMAND ----------

comp_df_vip=pd.merge(df_vip_summarized,df_vip_test_summarized,on='customer_id',how='inner')

# COMMAND ----------

comp_df_vip.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cross tab

# COMMAND ----------

filtered_comp_df_vip = comp_df_vip[(comp_df_vip['predicted_purchases'] >= 1) & 
                                  (comp_df_vip['predicted_purchases'] <= 7) &
                                  (comp_df_vip['actual_visits'] >= 1) &
                                   (comp_df_vip['actual_visits'] <= 7)]

# COMMAND ----------

cross_tab = pd.crosstab(filtered_comp_df_vip['predicted_purchases'], filtered_comp_df_vip['actual_visits'])

# COMMAND ----------

diagonal_sum = np.trace(cross_tab)
total_sum = np.sum(cross_tab).sum()
diagonal_accuracy = diagonal_sum / total_sum
print(f'Diagonal Accuracy:{diagonal_accuracy}') 
close_values_sum_vip = np.sum(np.diagonal(cross_tab, offset=1))
first_way_accuracy = (diagonal_sum + close_values_sum_vip )/ total_sum
print("One Increment Accuracy:", first_way_accuracy)
selected_cross_tab = cross_tab[3:]
lower_sum = np.sum(np.diagonal(selected_cross_tab, offset=-1))
upper_sum = np.sum(np.diagonal(selected_cross_tab, offset=1))
print("Quartile Accuracy:", (diagonal_sum+ lower_sum+upper_sum)/total_sum )
close_values_sum_vip1 = np.sum(np.diagonal(cross_tab, offset=1))  +np.sum(np.diagonal(cross_tab, offset=2)) 
second_way_accuracy = (diagonal_sum + close_values_sum_vip1 )/ total_sum
print("Two Increments Accuracy:", second_way_accuracy)

# COMMAND ----------

crosstab_percentage = cross_tab.div(cross_tab.sum(axis=0), axis=1) * 100

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(crosstab_percentage, annot=True, cmap="YlGnBu", fmt=".1f", cbar=True)
plt.title('Predicted Visits Distribution for VIP Customers in May (%)')
plt.xlabel('Actual Frequencies')
plt.ylabel('Predicted Frequencies')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Flags 

# COMMAND ----------

vip_flagged = comp_df_vip[['customer_id','predicted_purchases','actual_visits' ]]

# COMMAND ----------

def determine_flag(predicted, actual):
    if actual >= predicted:
        return 'Green'
    elif actual <= 0.5 * predicted:
        return 'Red'
    else:
        return 'Orange'

# COMMAND ----------

vip_flagged['Flag'] = vip_flagged.apply(
    lambda row: determine_flag(row['predicted_purchases'], row['actual_visits']), axis=1
)

# COMMAND ----------

grouped_data = vip_flagged.groupby('Flag').agg(customer_count = ('customer_id','count')).reset_index()
zero_predicted_customers = vip_flagged[vip_flagged['predicted_purchases'] == 0]['customer_id'].nunique()
grouped_data = grouped_data.append({'Flag': 'Zero Predicted Purchases', 'customer_count': zero_predicted_customers}, ignore_index=True)

plt.figure(figsize=(10, 6))
custom_palette = {"Green": "green", "Orange": "orange", "Red": "red", "Zero Predicted Purchases": "blue"}
ax = sns.barplot(x='Flag', y='customer_count', data=grouped_data, palette=custom_palette)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
sns.barplot(x='Flag', y='customer_count', data=grouped_data, palette= custom_palette)
plt.title('VIP Customer Count by Flag')
plt.xlabel('Flag')
plt.ylabel('Customer Count')
plt.show()

# COMMAND ----------

vip_flagged[vip_flagged['Flag'] == 'Red'][:10]

# COMMAND ----------

vip_flagged[vip_flagged['Flag'] == 'Green'][:10]

# COMMAND ----------

# MAGIC %md
# MAGIC #Frequentist

# COMMAND ----------

# MAGIC %md
# MAGIC ##Frequentist Transactional Data

# COMMAND ----------

sdf1=spark.sql("""select customer_id,transaction_id,transaction_date, transaction_value from sandbox.ps_trans_data_bdnbg where segment='Frequentist'""")
df_freq = sdf1.toPandas()

# COMMAND ----------

df_freq.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Splitting into test and train

# COMMAND ----------

train_data1 = df_freq[df_freq['transaction_date'] < '2024-05-01'] 
test_data1 = df_freq[(df_freq['transaction_date'] >= '2024-05-01') & (df_freq['transaction_date'] <= '2024-05-31')] 

print("Start Train dataset date {}".format(train_data1["transaction_date"].min()))
print("End Train dataset date {}".format(train_data1["transaction_date"].max()))
print("---------------------------------------------")
print("Start Test dataset date {}".format(test_data1["transaction_date"].min()))
print("End Test dataset date {}".format(test_data1["transaction_date"].max()))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Summarizing Transactional Data

# COMMAND ----------

#This function generates the recency, frequency, monetary values and tenure for the given train data customer IDs
df_freq_summarized = summary_data_from_transaction_data(train_data1, 
                                         'customer_id', 
                                         'transaction_date', 
                                         'transaction_value',
                                         observation_period_end='2024-04-30')
df_freq_summarized.reset_index(level=0, inplace=True)
df_freq_summarized.shape

# COMMAND ----------

df_freq_summarized.head()

# COMMAND ----------

df_freq_summarized['frequency'].describe(percentiles= [0.99])

# COMMAND ----------

df_freq_summarized['recency'].describe(percentiles= [0.99])

# COMMAND ----------

# MAGIC %md
# MAGIC ##Plots

# COMMAND ----------

plt.hist(df_freq_summarized['recency'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Recency')
plt.title('Distribution of Recency-Frequentist')
plt.show()
plt.savefig('recency.png')

# COMMAND ----------

sns.histplot(df_freq_summarized['frequency'], binwidth=10, color= 'red', edgecolor='black')
plt.xlabel('frequency')
plt.title('Distribution of frequency-Frequentist')
plt.show()

# COMMAND ----------

plt.hist(df_freq_summarized['monetary_value'], bins=30, edgecolor='k',range=(0, 1000))
plt.title('Spend Distribution-Frequentist')
plt.xlabel('Monetary Value')
plt.xlim(0, 1000)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Fitting the data to the model

# COMMAND ----------

bgf1 = BetaGeoFitter(penalizer_coef=0.1)
bgf1.fit(df_freq_summarized['frequency'], df_freq_summarized['recency'], df_freq_summarized['T'])
bgf1.summary

# COMMAND ----------

# MAGIC %md
# MAGIC ##Predicting Visits for the next month

# COMMAND ----------

t = 30 #30 days
df_freq_summarized['predicted_purchases'] = bgf1.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                                      df_freq_summarized['frequency'], 
                                                                                      df_freq_summarized['recency'], 
                                                                                      df_freq_summarized['T'])

# COMMAND ----------

df_freq_summarized['predicted_purchases'] = np.round(df_freq_summarized['predicted_purchases'],0)

# COMMAND ----------

df_freq_summarized[['customer_id','predicted_purchases']].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Prediciting Monetary Value(Revenue) for the next month

# COMMAND ----------

df_freq_summarized_positive = df_freq_summarized[df_freq_summarized['monetary_value'] > 0]

# COMMAND ----------

ggf1 = GammaGammaFitter(penalizer_coef = 0.1)
ggf1.fit(df_freq_summarized_positive['frequency'],
        df_freq_summarized_positive['monetary_value'])  
print(ggf1)

# COMMAND ----------

df_freq_summarized["predicted_amount"] = ggf1.customer_lifetime_value(
    bgf1, #the model to use to predict the number of future transactions
    df_freq_summarized['frequency'],
    df_freq_summarized['recency'],
    df_freq_summarized['T'],
    df_freq_summarized['monetary_value'],
    time= 1,
    discount_rate=0.01 # monthly discount rate
)

# COMMAND ----------

df_freq_summarized.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Plot

# COMMAND ----------

plt.hist(df_freq_summarized['predicted_purchases'], bins=20, color= 'red', edgecolor='black')
plt.xlabel('Number of Visits')
plt.title('Distribution of VIPs Predicted Number of Visits-Frequentist')
plt.show()

# COMMAND ----------

plt.hist(df_freq_summarized['predicted_amount'], bins=30, edgecolor='k',range=(0, 1000))
plt.title('Predicted amount Distribution-Frequentist')
plt.xlabel('Spend')
plt.xlim(0, 1000)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Getting Actual Visits

# COMMAND ----------

df_visits1 = test_data1.groupby('customer_id').nunique()['transaction_date'].reset_index()
df_visits1.rename(columns={'transaction_date': 'actual_visits'}, inplace=True)

# COMMAND ----------

df_visits1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Getting Actual Spend

# COMMAND ----------

df_amount1 = (test_data1.groupby(["customer_id"])[["transaction_value"]].agg("sum").reset_index().rename(columns={"transaction_value": "actual_amount"}))

# COMMAND ----------

df_amount1.head()

# COMMAND ----------

# Merge the two DataFrames of actual visits and actual amount
df_freq_test_summarized = pd.merge(df_visits1, df_amount1, on='customer_id')

# COMMAND ----------

df_freq_test_summarized.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Merging Predicted and Actual visits

# COMMAND ----------

comp_df_freq=pd.merge(df_freq_summarized,df_freq_test_summarized,on='customer_id',how='inner')

# COMMAND ----------

comp_df_freq.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cross Tab

# COMMAND ----------

filtered_comp_df_freq = comp_df_freq[(comp_df_freq['predicted_purchases'] >= 1) & 
                                  (comp_df_freq['predicted_purchases'] <= 7) &
                                  (comp_df_freq['actual_visits'] >= 1) &
                                   (comp_df_freq['actual_visits'] <= 7)]

# COMMAND ----------

cross_tab1 = pd.crosstab(filtered_comp_df_freq['predicted_purchases'], filtered_comp_df_freq['actual_visits'])

# COMMAND ----------

cross_tab1 = pd.crosstab(filtered_comp_df_freq['predicted_purchases'], filtered_comp_df_freq['actual_visits'])
cross_tab1
diagonal_sum1 = np.trace(cross_tab1)
total_sum1 = np.sum(cross_tab1).sum()
diagonal_accuracy1 = diagonal_sum1 / total_sum1
print(f'Diagonal Accuracy:{diagonal_accuracy1}') 
close_values_sum_freq = np.sum(np.diagonal(cross_tab1, offset=1))
first_way_accuracy1 = (diagonal_sum1 + close_values_sum_freq )/ total_sum1
print("One Increment Accuracy:", first_way_accuracy1)
selected_cross_tab1 = cross_tab1[3:]
lower_sum1 = np.sum(np.diagonal(selected_cross_tab1, offset=-1))
upper_sum1 = np.sum(np.diagonal(selected_cross_tab1, offset=1))
print("Quartile Accuracy:", (diagonal_sum1+ lower_sum1+upper_sum1)/total_sum1 )
close_values_sum_freq1 = np.sum(np.diagonal(cross_tab1, offset=1))  +np.sum(np.diagonal(cross_tab1, offset=2)) 
second_way_accuracy1 = (diagonal_sum1 + close_values_sum_freq1 )/ total_sum1
print("Two Increments Accuracy:", second_way_accuracy1)

# COMMAND ----------

crosstab_percentage1 = cross_tab1.div(cross_tab1.sum(axis=0), axis=1) * 100

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(crosstab_percentage1, annot=True, cmap="YlGnBu", fmt=".1f", cbar=True)
plt.title('Predicted Visits Distribution for Frequentist Customers in May (%)')
plt.xlabel('Actual Frequencies')
plt.ylabel('Predicted Frequencies')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Flags

# COMMAND ----------

freq_flagged = comp_df_freq[['customer_id','predicted_purchases','actual_visits' ]]

# COMMAND ----------

def determine_flag(predicted, actual):
    if actual >= predicted:
        return 'Green'
    elif actual <= 0.5 * predicted:
        return 'Red'
    else:
        return 'Orange'

# COMMAND ----------

freq_flagged['Flag'] = freq_flagged.apply(
    lambda row: determine_flag(row['predicted_purchases'], row['actual_visits']), axis=1
)

# COMMAND ----------

grouped_data1 = freq_flagged.groupby('Flag').agg(customer_count = ('customer_id','count')).reset_index()
zero_predicted_customers1 = freq_flagged[freq_flagged['predicted_purchases'] == 0]['customer_id'].nunique()
grouped_data1 = grouped_data1.append({'Flag': 'Zero Predicted Purchases', 'customer_count': zero_predicted_customers1}, ignore_index=True)

plt.figure(figsize=(10, 6))
custom_palette = {"Green": "green", "Orange": "orange", "Red": "red", "Zero Predicted Purchases": "blue"}
ax = sns.barplot(x='Flag', y='customer_count', data=grouped_data1, palette=custom_palette)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
sns.barplot(x='Flag', y='customer_count', data=grouped_data1, palette= custom_palette)
plt.title('Frequentist Customer Count by Flag')
plt.xlabel('Flag')
plt.ylabel('Customer Count')
plt.show()

# COMMAND ----------

freq_flagged[freq_flagged['Flag'] == 'Red'][:10]

# COMMAND ----------

freq_flagged[freq_flagged['Flag'] == 'Red'][:10]

# COMMAND ----------

# MAGIC %md
# MAGIC #Moderate

# COMMAND ----------

# MAGIC %md
# MAGIC ##Moderate Transactional Data

# COMMAND ----------

sdf2=spark.sql("""select customer_id,transaction_id,transaction_date, transaction_value from sandbox.ps_trans_data_bdnbg where segment='Moderate'""")
df_mod = sdf2.toPandas()

# COMMAND ----------

df_mod.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Splitting into train and test

# COMMAND ----------

train_data2 = df_mod[df_mod['transaction_date'] < '2024-05-01'] 
test_data2 = df_mod[(df_mod['transaction_date'] >= '2024-05-01') & (df_mod['transaction_date'] <= '2024-05-31')] 

print("Start Train dataset date {}".format(train_data2["transaction_date"].min()))
print("End Train dataset date {}".format(train_data2["transaction_date"].max()))
print("---------------------------------------------")
print("Start Test dataset date {}".format(test_data2["transaction_date"].min()))
print("End Test dataset date {}".format(test_data2["transaction_date"].max()))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Summarizing Transactional Data

# COMMAND ----------

#This function generates the recency, frequency, monetary values and tenure for the given train data customer IDs
df_mod_summarized = summary_data_from_transaction_data(train_data2, 
                                         'customer_id', 
                                         'transaction_date', 
                                         'transaction_value',
                                         observation_period_end='2024-04-30')
df_mod_summarized.reset_index(level=0, inplace=True)
df_mod_summarized.shape

# COMMAND ----------

df_mod_summarized.head()

# COMMAND ----------

df_mod_summarized['recency'].describe(percentiles= [0.99])

# COMMAND ----------

df_mod_summarized['frequency'].describe(percentiles= [0.99])

# COMMAND ----------

# MAGIC %md
# MAGIC ##Plots

# COMMAND ----------

plt.hist(df_mod_summarized['recency'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Recency')
plt.title('Distribution of Recency-Moderate')
plt.show()
plt.savefig('recency.png')

# COMMAND ----------

plt.hist(df_mod_summarized['frequency'], bins=20, color='red', edgecolor='black')
plt.xlabel('Recency')
plt.title('Distribution of Frequency-Moderate')
plt.show()
plt.savefig('recency.png')

# COMMAND ----------

plt.hist(df_mod_summarized['monetary_value'], bins=30, edgecolor='k',range=(0, 1000))
plt.title('Spend Distribution-Moderate')
plt.xlabel('Monetary Value')
plt.xlim(0, 1000)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Fitting the data to the model

# COMMAND ----------

bgf2 = BetaGeoFitter(penalizer_coef=0.1)
bgf2.fit(df_mod_summarized['frequency'], df_mod_summarized['recency'], df_mod_summarized['T'])
bgf2.summary

# COMMAND ----------

# MAGIC %md
# MAGIC ##Predicting Visits for the next month

# COMMAND ----------

t = 30 #30 days
df_mod_summarized['predicted_purchases'] = bgf2.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                                      df_mod_summarized['frequency'], 
                                                                                      df_mod_summarized['recency'], 
                                                                                      df_mod_summarized['T'])

# COMMAND ----------

df_mod_summarized['predicted_purchases'] = np.round(df_mod_summarized['predicted_purchases'],0)

# COMMAND ----------

df_mod_summarized[['customer_id','predicted_purchases']].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Predicting Monetary Value(Revenue) for the next month

# COMMAND ----------

df_mod_summarized_positive = df_mod_summarized[df_mod_summarized['monetary_value'] > 0]

# COMMAND ----------

ggf2 = GammaGammaFitter(penalizer_coef = 0.1)
ggf2.fit(df_mod_summarized_positive['frequency'],
        df_mod_summarized_positive['monetary_value'])  
print(ggf2)

# COMMAND ----------

df_mod_summarized["predicted_amount"] = ggf2.customer_lifetime_value(
    bgf2, #the model to use to predict the number of future transactions
    df_mod_summarized['frequency'],
    df_mod_summarized['recency'],
    df_mod_summarized['T'],
    df_mod_summarized['monetary_value'],
    time= 1,
    discount_rate=0.01 # monthly discount rate
)

# COMMAND ----------

df_mod_summarized.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Plot

# COMMAND ----------

plt.hist(df_mod_summarized['predicted_purchases'], bins=20, color= 'red', edgecolor='black')
plt.xlabel('Number of Visits')
plt.title('Distribution of VIPs Predicted Number of Visits-Moderate')
plt.show()

# COMMAND ----------

plt.hist(df_mod_summarized['predicted_amount'], bins=30, edgecolor='k',range=(0, 1000))
plt.title('Predicted amount Distribution-Moderate')
plt.xlabel('Spend')
plt.xlim(0, 1000)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Getting Actual Visits

# COMMAND ----------

df_visits2 = test_data2.groupby('customer_id').nunique()['transaction_date'].reset_index()
df_visits2.rename(columns={'transaction_date': 'actual_visits'}, inplace=True)

# COMMAND ----------

df_visits2.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Getting Actual Spend

# COMMAND ----------

df_amount2 = (test_data2.groupby(["customer_id"])[["transaction_value"]].agg("sum").reset_index().rename(columns={"transaction_value": "actual_amount"}))

# COMMAND ----------

df_amount2.head()

# COMMAND ----------

# Merge the two DataFrames of actual visits and actual amount
df_mod_test_summarized = pd.merge(df_visits2, df_amount2, on='customer_id')

# COMMAND ----------

df_mod_test_summarized.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Merging Predicted and Actual Visits

# COMMAND ----------

comp_df_mod=pd.merge(df_mod_summarized,df_mod_test_summarized,on='customer_id',how='inner')

# COMMAND ----------

comp_df_mod.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cross Tab

# COMMAND ----------

filtered_comp_df_mod = comp_df_mod[(comp_df_mod['predicted_purchases'] >= 1) & 
                                  (comp_df_mod['predicted_purchases'] <= 7) &
                                  (comp_df_mod['actual_visits'] >= 1) &
                                   (comp_df_mod['actual_visits'] <= 7)]

# COMMAND ----------

cross_tab2 = pd.crosstab(filtered_comp_df_mod['predicted_purchases'], filtered_comp_df_mod['actual_visits'])

# COMMAND ----------

cross_tab2 = pd.crosstab(filtered_comp_df_mod['predicted_purchases'], filtered_comp_df_mod['actual_visits'])
cross_tab2
diagonal_sum2 = np.trace(cross_tab2)
total_sum2 = np.sum(cross_tab2).sum()
diagonal_accuracy2 = diagonal_sum2 / total_sum2
print(f'Diagonal Accuracy:{diagonal_accuracy2}') 
close_values_sum_mod = np.sum(np.diagonal(cross_tab2 ,offset=1))
first_way_accuracy2 = (diagonal_sum2 + close_values_sum_mod )/ total_sum2
print("One Increment Accuracy:", first_way_accuracy2)
selected_cross_tab2 = cross_tab2[3:]
lower_sum2 = np.sum(np.diagonal(selected_cross_tab2, offset=-1))
upper_sum2 = np.sum(np.diagonal(selected_cross_tab2, offset=1))
print("Quartile Accuracy:", (diagonal_sum2+ lower_sum2+upper_sum2)/total_sum2 )
close_values_sum_mod1 = np.sum(np.diagonal(cross_tab2, offset=1))  +np.sum(np.diagonal(cross_tab2, offset=2)) 
second_way_accuracy2 = (diagonal_sum2 + close_values_sum_mod1 )/ total_sum2
print("Two Increments Accuracy:", second_way_accuracy2)

# COMMAND ----------



# COMMAND ----------

crosstab_percentage2 = cross_tab2.div(cross_tab2.sum(axis=0), axis=1) * 100

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(crosstab_percentage2, annot=True, cmap="YlGnBu", fmt=".1f", cbar=True)
plt.title('Predicted Visits Distribution for Frequentist Customers in May (%)')
plt.xlabel('Actual Frequencies')
plt.ylabel('Predicted Frequencies')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Flags

# COMMAND ----------

mod_flagged = comp_df_mod[['customer_id','predicted_purchases','actual_visits' ]]

# COMMAND ----------

def determine_flag(predicted, actual):
    if actual >= predicted:
        return 'Green'
    elif actual <= 0.5 * predicted:
        return 'Red'
    else:
        return 'Orange'

# COMMAND ----------

mod_flagged['Flag'] = mod_flagged.apply(
    lambda row: determine_flag(row['predicted_purchases'], row['actual_visits']), axis=1
)

# COMMAND ----------

grouped_data2 = mod_flagged.groupby('Flag').agg(customer_count = ('customer_id','count')).reset_index()
zero_predicted_customers2 = mod_flagged[mod_flagged['predicted_purchases'] == 0]['customer_id'].nunique()
grouped_data2 = grouped_data2.append({'Flag': 'Zero Predicted Purchases', 'customer_count': zero_predicted_customers2}, ignore_index=True)

plt.figure(figsize=(10, 6))
custom_palette = {"Green": "green", "Orange": "orange", "Red": "red", "Zero Predicted Purchases": "blue"}
ax = sns.barplot(x='Flag', y='customer_count', data=grouped_data2, palette=custom_palette)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
sns.barplot(x='Flag', y='customer_count', data=grouped_data2, palette= custom_palette)
plt.title('Moderate Customer Count by Flag')
plt.xlabel('Flag')
plt.ylabel('Customer Count')
plt.show()

# COMMAND ----------

mod_flagged[mod_flagged['Flag'] == 'Red'][:10]

# COMMAND ----------

mod_flagged[mod_flagged['Flag'] == 'Green'][:10]

# COMMAND ----------

spark_df = spark.createDataFrame(vip_flagged)
spark_df.write.option("overwriteSchema", "true").mode("overwrite").saveAsTable("sandbox.am_vipflagged_may")
spark_df = spark.createDataFrame(freq_flagged)
spark_df.write.option("overwriteSchema", "true").mode("overwrite").saveAsTable("sandbox.am_freqflagged_may")
spark_df = spark.createDataFrame(mod_flagged)
spark_df.write.option("overwriteSchema", "true").mode("overwrite").saveAsTable("sandbox.am_modflagged_may")

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table sandbox.customer_visits_may
# MAGIC (
# MAGIC   select * from sandbox.am_vipflagged_may
# MAGIC   UNION 
# MAGIC   select * from sandbox.am_freqflagged_may
# MAGIC   UNION 
# MAGIC   select * from sandbox.am_modflagged_may
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sandbox.customer_visits_may
# MAGIC
