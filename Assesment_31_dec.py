import pandas as pd
import hvplot.pandas
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import io
import duckdb 
import sqlalchemy
import os
import numpy as np
import pyarrow.parquet as pq
from io import BytesIO, StringIO
import requests
from holoviews import save
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bokeh.models.formatters import PrintfTickFormatter
from sklearn.preprocessing import LabelEncoder



def telegram_send_photo(caption, x):
    current_date = datetime.now().strftime('%d-%m-%Y')
    bot_token = ''
    chat_id = ''  # Replace with the chat ID where you want to send the image

    #caption = f'Total Donation for every Year for all type of {K} {current_date}.png'

    # URL for sending a photo to the bot
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'

    files = {'photo':  ('Image_blood_percent.png', x)}
    data = {'chat_id': chat_id, 'caption': caption }

    response = requests.post(url, files=files, data=data)

    # Check the response
    if response.status_code == 200:
        print('Image sent successfully!')
    else:
        print(f'Error {response.status_code}: {response.text}')

    
def send_document(y):   
    current_date = datetime.now().strftime('%d-%m-%Y')

    #plt.show()
    bot_token = ''
    chat_id = ''  # Replace with the chat ID where you want to send the document

    caption_classification_report = f'Classification report {current_date}.txt'

    # URL for sending a document to the bot
    url_doc = f'https://api.telegram.org/bot{bot_token}/sendDocument'

    files_class_report = {'document': ('classification_report.txt',y)}
    data_class_report = {'chat_id': chat_id, 'caption': caption_classification_report}

    response = requests.post(url_doc, files=files_class_report, data=data_class_report)

    # Check the response
    if response.status_code == 200:
        print('Document sent successfully!')
    else:
        print(f'Error {response.status_code}: {response.text}')



def data_machine_learning_report():
    
    def read_parquet():
        url = "https://dub.sh/ds-data-granular"

    # Make a request to the URL and read the Parquet file
        response = requests.get(url)
        parquet_data = BytesIO(response.content)
        df = pq.read_table(parquet_data).to_pandas()
        #print(df.tail(6))
        return df

    df = read_parquet()
    df['visit_date'] = pd.to_datetime(df['visit_date'])

    df['recency'] = (pd.to_datetime('today') - df['visit_date']).dt.days

    #Churn out is defined by people that Average last donate day more than 2 years 
    churn_threshold = 730
    df['churn'] = (df['recency'] > churn_threshold).astype(int)

    #df

    df_1 = duckdb.sql("""
        SELECT 
            donor_id,
            birth_date,
            ROUND(AVG(recency),2) as 'Average_donate_day_from_recent_day',
            churn
        FROM df
        
        WHERE birth_date > 1964 
        GROUP BY 
            donor_id,
            birth_date,
            churn            
                   
    """).df()

    print(df_1)

    #df_1 = data_machine_learning_report()

    features = ['birth_date', 'Average_donate_day_from_recent_day']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df_1[features], df_1['churn'], test_size=0.2, random_state=10)

    model = xgb.XGBClassifier()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    print("Classification Report:")
    class_report = classification_report(y_test, y_pred)
    print(class_report)

    #np.savetxt('classification_report.txt', [class_report], fmt='%s')

    output_file = io.BytesIO()
    output_file.write(class_report.encode())

    # Create BytesIO object for the text content

    # Reset the buffer position to the beginning
    output_file.seek(0)

    send_document(output_file)  

    #Display confusion matrix and classification report

    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    print("\nConfusion Matrix (Percentage):")
    print(conf_matrix_percentage)


    sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Percentage)')


    conf_matrix_percentage = io.BytesIO()
    plt.savefig(conf_matrix_percentage, format='png', dpi=300, bbox_inches='tight')

    # Close the plot to prevent any interference
    #plt.close()

    # Reset the buffer position to the beginning
    conf_matrix_percentage.seek(0)

    current_date = datetime.now().strftime('%d-%m-%Y')


    #plt.show()
    #plt.close()
    caption_confusion_matrix = f'Confusuion matrix {current_date}.png'
    telegram_send_photo(caption_confusion_matrix,conf_matrix_percentage)



    #single_observation = pd.DataFrame({'birth_date': [2000], 'Average_donate_day_from_recent_day': [5]})
    prediction = model.predict(df_1[features])

    # Print the prediction
    print(f'Churn Prediction: {prediction}')

    
data_machine_learning_report()

def telegram_send_photo(caption, x):
    current_date = datetime.now().strftime('%d-%m-%Y')
    bot_token = '6627132548:AAFjZNcqGUfqSFP2No07LQO_YasAKTcNP7M'
    chat_id = '-4199876403'  # Replace with the chat ID where you want to send the image

    #caption = f'Total Donation for every Year for all type of {K} {current_date}.png'

    # URL for sending a photo to the bot
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'

    files = {'photo':  ('Image_blood_percent.png', x)}
    data = {'chat_id': chat_id, 'caption': caption }

    response = requests.post(url, files=files, data=data)

    # Check the response
    if response.status_code == 200:
        print('Image sent successfully!')
    else:
        print(f'Error {response.status_code}: {response.text}')


def donor_facility():

    def read_csv_don_facility():
        url_csv_1 = "https://raw.githubusercontent.com/MoH-Malaysia/data-darah-public/main/donations_facility.csv"


        df_1 = pd.read_csv(url_csv_1)
        return df_1

    df_don_facility = read_csv_don_facility()
   
    df_don_facility['date'] = pd.to_datetime(df_don_facility['date'])
    
    df_don_facility= duckdb.sql("""
                 
    SELECT 
        YEAR(date) as Year, hospital, daily,	blood_a,	blood_b,	blood_o,	blood_ab,	location_centre,	location_mobile,	type_wholeblood,type_apheresis_platelet,type_apheresis_plasma,	type_other,	social_civilian,social_student,	social_policearmy,	donations_new,	donations_regular,	donations_irregular
        
    FROM df_don_facility

        """).df()

    df_don_facility_year= duckdb.sql("""
                 
    SELECT 
        Year, 
        SUM(CAST ("daily" AS DOUBLE)) as 'Total donation',
        --SUM(CAST(("blood_a" +"blood_b" + "blood_o" + "blood_ab") as DOUBLE)) as Total_type_of_blood,
        ROUND((SUM(CAST ("blood_a" as DOUBLE))/SUM(CAST ("daily" AS DOUBLE)))*100,1) as blood_a, 
        ROUND((SUM(CAST ("blood_b" AS DOUBLE))/SUM(CAST ("daily" AS DOUBLE)))*100 ,1)as blood_b,
        ROUND((SUM(CAST ("blood_o" AS DOUBLE))/SUM(CAST ("daily" AS DOUBLE)))*100,1) as blood_o,
        ROUND((SUM(CAST ("blood_ab" AS DOUBLE))/SUM(CAST ("daily" AS DOUBLE)))*100,1) as blood_ab                                
    FROM df_don_facility                
    GROUP BY  Year,
    ORDER BY  Year ASC

        """).df()

    #print(df_don_facility_year)
        

    #print(df_don_facility)

    #df_don_facility_year_1 = df_don_facility_year['Total donation']/1000
    df_don_facility_year_1 = df_don_facility_year.assign(Total_donation=lambda x: x['Total donation'] / 1000)
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Year', y='Total donation', data=df_don_facility_year_1, marker='o', color='blue')

    def format_func(value, tick_number):
        return f'{int(value)}'

    def format_func_y(value, tick_number):
        return f'{int(round(value / 1000))}'

    for index, row in df_don_facility_year_1.iterrows():
        plt.annotate(f'{int(round(row["Total donation"]/1000))}', (row['Year'], row['Total donation']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

    plt.ylabel('Total donation in thousand')

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func_y))

    #line_chart_total_people.opts(yformatter=PrintfTickFormatter(format='%f'))

    plt.title("Total Donation for every Year for any blood type",linespacing=1.8)

    current_date = datetime.now().strftime('%d-%m-%Y')
    filename_blood_total = f'Total donation plot for all {current_date}.png'

    plt.title("Total Donation for every Year for all type of blood",linespacing=1.8)
    plt.savefig(filename_blood_total , dpi=300, bbox_inches='tight')

    image_data_blood_total = io.BytesIO()
    plt.savefig(image_data_blood_total, format='png')
    image_data_blood_total.seek(0)

    #plt.show()
    #plt.close()

    
    caption_image_data_blood_total = f'Total Donation for all blood from 2006-2024 {current_date}.png'
 
    telegram_send_photo(caption_image_data_blood_total,image_data_blood_total)



    df_blood = df_don_facility_year[['Year','blood_a','blood_b','blood_o','blood_ab']]
    plt.figure(figsize=(12, 8))
    plt.title("Percentage Blood donor for every year")

    plt.legend("Type of blood")

    plt.xticks(df_blood['Year'])

    def format_func(value, tick_number):
        return f'{int(value)}'

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

    plt.ylabel("Percentage %")

    sns.lineplot(x='Year', y='blood_a', data=df_blood, marker='o', color='red', label='blood_a')

    # Plotting blood_b
    sns.lineplot(x='Year', y='blood_b', data=df_blood, marker='o', color='blue', label='blood_b')

    # Plotting blood_o
    sns.lineplot(x='Year', y='blood_o', data=df_blood, marker='o', color='green', label='blood_o')

    # Plotting blood_ab
    sns.lineplot(x='Year', y='blood_ab', data=df_blood, marker='o', color='purple', label='blood_ab')

    for index, row in df_blood.iterrows():
        plt.annotate(f'{float(row["blood_a"])}', (row['Year'], row['blood_a']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')
        
    for index, row in df_blood.iterrows():
        plt.annotate(f'{float(row["blood_b"])}', (row['Year'], row['blood_b']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')
    for index, row in df_blood.iterrows():
        plt.annotate(f'{float(row["blood_o"])}', (row['Year'], row['blood_o']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')
    for index, row in df_blood.iterrows():
        plt.annotate(f'{float(row["blood_ab"])}', (row['Year'], row['blood_ab']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')
        

    current_date = datetime.now().strftime('%d-%m-%Y')
    filename_blood_percent = f'Percentage donation plot for all {current_date}.png'

    plt.title("Percentage Donation for every Year for all type of blood",linespacing=1.8)
    plt.savefig(filename_blood_percent , dpi=300, bbox_inches='tight')

    image_data_blood_percent = io.BytesIO()
    plt.savefig(image_data_blood_percent, format='png')
    image_data_blood_percent.seek(0)

    #plt.show()
    #plt.close()

    caption_percentage_all_blood = f'Percentage donation plot for all type of blood {current_date}.png'
 
    telegram_send_photo(caption_percentage_all_blood,image_data_blood_percent)

    #telegram_send_photo(image_data_blood_percent)

donor_facility()


def newdonor_facility():

    def read_csv_new_don_facility():
        url_csv_3= "https://raw.githubusercontent.com/MoH-Malaysia/data-darah-public/main/newdonors_facility.csv"
        df_3 = pd.read_csv(url_csv_3)

   
        return df_3

    df_3 = read_csv_new_don_facility()
    df_3['date'] = pd.to_datetime(df_3['date'])
    
    df_3_extract = duckdb.sql("""
    SELECT 
        YEAR(date) as Year,
        SUM("17-24") as "17_to_24",
        SUM("25-29") as "25_to_29",
        SUM("30-34") as "30_to_34",
        SUM("35-39") as "35_to_39",
        SUM("40-44") as "40_to_44",
        SUM("45-49") as "45_to_49",
        SUM("50-54") as "50_to_54",
        SUM("55-59") as "55_to_59",
        SUM("60-64") as "60_to_64",
        SUM(other) as "other",
        hospital,
        SUM("total") AS  total 
    FROM df_3
    GROUP BY Year, hospital 
    ORDER BY Year ASC
    """).df()
    df_3_extract



    df_total_age_total = duckdb.sql("""
                    
    WITH table_total_age as(           
        SELECT 
            SUM("17_to_24"+"25_to_29"+"30_to_34"+"35_to_39"+"40_to_44"+"45_to_49"+"50_to_54"+"55_to_59"+"60_to_64"+"other") as Total_People,
            --SUM("total") as total,
            Year,
            SUM("17_to_24") as "Total_17-24",
            SUM("25_to_29") as "Total_25-29",
            sum("30_to_34") as "Total_30-34",
            sum("35_to_39") as "Total_35-39",
            sum("40_to_44") as "Total_40-44",
            sum("45_to_49") as "Total_45-49",
            sum("50_to_54") as "Total_50-54",
            sum("55_to_59") as "Total_55-59",
            sum("60_to_64") as "Total_60-64",
            sum("other") as "other"
                
        FROM 
            df_3_extract 
        GROUP BY Year
        Order By Year
            )
                                
    SELECT 
        SUM("Total_People") as "Total_People",
        SUM("Total_17-24") as "Total_17-24",
        SUM("Total_25-29") as "Total_25-29",
        sum("Total_30-34") as "Total_30-34",
        sum("Total_35-39") as "Total_35-39",
        sum("Total_40-44") as "Total_40-44",
        sum("Total_45-49") as "Total_45-49",
        sum("Total_50-54") as "Total_50-54",
        sum("Total_55-59") as "Total_55-59",
        sum("Total_60-64") as "Total_60-64",
        sum("other") as "other"
        
                                
    FROM 
        table_total_age 


    """).df()
    df_total_age_total
    df_long = df_total_age_total.melt(value_vars=df_total_age_total.columns[:], var_name='Age Group', value_name='Total')

    plt.figure(figsize=(20, 8))

    ax = sns.barplot(x = df_long['Total'] , y=df_long['Age Group'], orient='h', color = 'red') 

    sns.despine(bottom=True)
    #plt.set_xlabel('')
    plt.xticks([])
    plt.xlabel('')

    

    plt.title('Total People donor for evey age from 2006 - 2024 current in thousand', size = 20)

    
    for rect in ax.patches:
        ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, f'{float(round(rect.get_width())/1000)}',
                ha='left', va='center', fontsize=10, color='black')
        

    current_date = datetime.now().strftime('%Y%m%d')
    filename_df_long = f'File total donation for every age until current year in thousand{current_date}.png'

    plt.title("Total donation for every age until current year in thousand",linespacing=1.8, size = 20)
    plt.savefig(filename_df_long, dpi=300, bbox_inches='tight')

    image_data_age_total = io.BytesIO()
    plt.savefig(image_data_age_total, format='png')
    image_data_age_total.seek(0)

    current_date = datetime.now().strftime('%d-%m-%Y')

    #plt.show()
    #plt.close()

    caption_Total_people_donor_age = f'Total People donor for evey age from 2006 - 2024 {current_date}.png'
 
    telegram_send_photo(caption_Total_people_donor_age, image_data_age_total)

    #telegram_send_photo(image_data_age_total)


newdonor_facility()

def donor_facility_social():

    def read_csv_don_facility():
        url_csv_1 = "https://raw.githubusercontent.com/MoH-Malaysia/data-darah-public/main/donations_facility.csv"


        df_1 = pd.read_csv(url_csv_1)
        return df_1

    df_don_facility = read_csv_don_facility()
   
    df_don_facility['date'] = pd.to_datetime(df_don_facility['date'])
    
    df_don_facility= duckdb.sql("""
                 
    SELECT 
        YEAR(date) as Year, hospital, daily,	blood_a,	blood_b,	blood_o,	blood_ab,	location_centre,	location_mobile,	type_wholeblood,	type_apheresis_platelet,	type_apheresis_plasma,	type_other,	social_civilian,	social_student,	social_policearmy,	donations_new,	donations_regular,	donations_irregular
        
    FROM df_don_facility

        """).df()

    df_don_facility_social= duckdb.sql("""
                 
    SELECT 
        Year, 
        SUM(CAST ("daily" AS DOUBLE)) as 'Total donation',
        SUM(social_student) as social_student,
        SUM(social_policearmy) as social_policearmy,
        SUM(social_civilian) as social_civilian
                                       
    FROM df_don_facility                
    GROUP BY  Year,
    ORDER BY  Year ASC

        """).df()
    
    plt.figure(figsize=(20, 8))

    def format_func(value, tick_number):
        return f'{int(value)}'
    
    def format_func_y(value, tick_number):
        return f'{int(round(int(value) / 1000, 1))}'

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func_y))

    plt.title("Total donation for social type from 2006 to 2024", size = 18)
    
    
    print(df_don_facility_social)
    plt.ylabel("Total donation in thousand")
   

    # Plotting blood_b
    sns.lineplot(x='Year', y='social_student', data=df_don_facility_social, marker='o', color='blue', label='social_student')

    # Plotting blood_o
    sns.lineplot(x='Year', y='social_policearmy', data=df_don_facility_social, marker='o', color='black', label='social_policearmy')

    # Plotting blood_ab
    sns.lineplot(x='Year', y='social_civilian', data=df_don_facility_social, marker='o', color='red', label='social_civilian')

   
        
    for index, row in df_don_facility_social.iterrows():
        plt.annotate(f'{int(row["social_student"]/1000)}', (row['Year'], row['social_student']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='black')
        
    for index, row in df_don_facility_social.iterrows():
        plt.annotate(f'{int(row["social_policearmy"]/1000)}', (row['Year'], row['social_policearmy']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='black')
        
    for index, row in df_don_facility_social.iterrows():
        plt.annotate(f'{int(row["social_civilian"]/1000)}', (row['Year'], row['social_civilian']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='black')
        
    current_date = datetime.now().strftime('%d-%m-%Y')
    filename_social = f'Total donation plot for all social {current_date}.png'
    
    filename_social = io.BytesIO()
    plt.savefig(filename_social, format='png', dpi=300, bbox_inches='tight')
    filename_social.seek(0)
    
    caption_filename_social = f'Total donation for social police_army, civilians and student {current_date}.png'
    
    telegram_send_photo(caption_filename_social,filename_social)


donor_facility_social()
