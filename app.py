import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from scipy.stats import boxcox
from datetime import datetime
import smtplib

# Constants
BOXCOX_LAMBDA = 2.2851  # Box‐Cox λ from training
# PIPELINE_PATH = 'rf_pipeline.joblib'
DEFAULT_DATA_PATH = 'Customer_support_data.csv'
# At the top of your script (or inside load functions):
ct = joblib.load('preprocessor.joblib')        # fitted ColumnTransformer
xgb_model = joblib.load('xgb_model.joblib') # fitted XGBClassifier

api_key = "your_api_key_here"
if api_key:  # Only configure if the key is available
    genai.configure(api_key=api_key)
else:
    print("Warning: No API key provided. Some features may be disabled.")
genai_model = genai.GenerativeModel('models/gemini-1.5-flash')

@st.cache_data
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw or skip if already done."""
    # If already has target and transforms, skip
    if {'satisfaction_target','log_Response_Time','wait_time_boxcox'}.issubset(df.columns):
        return df

    # 1) Drop if exists
    for c in ['connected_handling_time']:
        if c in df: df.drop(columns=c, inplace=True)

    # 2) Basic cleaning
    df.dropna(subset=['Order_id'], inplace=True)
    def clean_text(text_series):
        cleaned = (
            text_series
            .str.lower()
            .str.replace(r'[^a-z\s]', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        # Replace blanks with NaN, then fill with 'no remarks'
        return cleaned.replace(r'^\s*$', np.nan, regex=True).fillna('no remarks')


    # Apply
    df['Customer Remarks'] = clean_text(df['Customer Remarks'])

    df['Product_category'].fillna('Not Available', inplace=True)
    df['Customer_City'].fillna('Not Given', inplace=True)

    # 3) Date parsing & feature extraction
    df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['issue_responded']    = pd.to_datetime(df['issue_responded'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['Survey_response_Date']=pd.to_datetime(df['Survey_response_Date'], format='%d-%b-%y', errors='coerce')
    df['order_date_time']    = pd.to_datetime(df['order_date_time'],  format='%d/%m/%Y %H:%M', errors='coerce')
    df['order_date_time']    = df.groupby('Survey_response_Date')['order_date_time'].transform(lambda x: x.fillna(x.median()))

    df['day']      = df['order_date_time'].dt.day_name()
    df['year']     = df['order_date_time'].dt.year
    df['month_num']= df['order_date_time'].dt.month
    df['day_num']  = df['order_date_time'].dt.day
    df['hour']     = df['order_date_time'].dt.hour
    df['minute']   = df['order_date_time'].dt.minute
    df['month']    = df['order_date_time'].dt.month_name()

    df['Response_Time_minutes'] = (
        df['issue_responded'] - df['Issue_reported at']
    ).dt.total_seconds()/60
    df = df[df['Response_Time_minutes']>=0]
    df['wait_time_minutes'] = (
        df['Issue_reported at'] - df['order_date_time']
    ).dt.total_seconds()/60

    for c in ['Issue_reported at','issue_responded','order_date_time','Survey_response_Date']:
        if c in df: df.drop(columns=c, inplace=True)

    # 4) Map experience
    mapping = {
        'On Job Training':'Beginner','0-30':'Beginner',
        '31-60':'Intermediate','61-90':'Intermediate',
        '>90':'Advanced'
    }
    df['Experience_Level'] = df.get('Tenure Bucket', pd.Series()).map(mapping).fillna('Beginner')
    if 'Tenure Bucket' in df: df.drop(columns='Tenure Bucket', inplace=True)
    df = df.drop(columns='Item_price')

    # 5) Target creation
    if 'CSAT Score' in df:
        df['satisfaction_target'] = df['CSAT Score'].apply(lambda x: 1 if pd.to_numeric(x,errors='coerce')>=4 else 0)
        for c in ['CSAT Score','Unique id']:
            if c in df: df.drop(columns=c, inplace=True)

    # 6) Clean numeric
    df.dropna(subset=['Response_Time_minutes','wait_time_minutes','satisfaction_target'], inplace=True)
    # df['Item_price'].fillna(0, inplace=True)

    # 7) Outlier removal
    def drop_iqr(df_in, cols, factor=1.5):
        dfc = df_in.copy()
        for col in cols:
            if col in dfc:
                q1,q3 = dfc[col].quantile([.25,.75])
                iqr = q3-q1
                dfc = dfc[(dfc[col]>=q1-factor*iqr)&(dfc[col]<=q3+factor*iqr)]
        return dfc
    df = drop_iqr(df, ['Response_Time_minutes','wait_time_minutes'])

    # 8) Transforms
    df['log_Response_Time'] = np.log1p(df['Response_Time_minutes'])
    df['wait_time_boxcox']  = boxcox(df['wait_time_minutes']+1e-6, BOXCOX_LAMBDA)

    for c in ['Response_Time_minutes','wait_time_minutes']:
        if c in df: df.drop(columns=c, inplace=True)

    return df

@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def generate_response(inputs: dict, prediction: int) -> str:
    """Generate a polite reply using Gemini."""
    prompt = (
        "You are a helpful customer service assistant.\n"
        "Based on the details below and the model's prediction, craft a concise, polite response.\n\n"
        f"Category: {inputs.get('category')} | Sub-category: {inputs.get('Sub-category')} | "
        f"City: {inputs.get('Customer_City')} "
        f"Experience Level: {inputs.get('Experience_Level')} | "
        f"Remarks: '{inputs.get('Customer Remarks')}'\n"
        f"Prediction: {'Satisfied' if prediction==1 else 'Unsatisfied'}\n"
        "Reply:"
    )
    chat = genai_model.start_chat()
    resp = chat.send_message(prompt)
    return resp.text.strip()

def main():
    st.title('Flipkart CSAT Predictor & Auto-Email & Chatbot')
    # Simulate a role check (you can later replace this with login auth)
    is_manager = st.sidebar.checkbox("Login as Manager")


    # 1) Data upload
    uploaded = st.sidebar.file_uploader('Upload CSV', type=['csv'], key='u1')
    if uploaded:
        df = preprocess_df(pd.read_csv(uploaded))
    else:
        st.sidebar.info('Using default data')
        df = preprocess_df(pd.read_csv(DEFAULT_DATA_PATH))

    # 2) Customer email
    customer_email = st.sidebar.text_input('Customer Email', key='email_inp')

    # 3) Prediction flow
    pipeline = load_pipeline()
    order_id = st.sidebar.selectbox('Order ID', df['Order_id'].unique(), key='o1')
    row = df[df['Order_id']==order_id]
    if st.sidebar.button('Predict & Send Email', key='b1'):
        # 1) transform the single-row DF into the numeric array
        X_numeric = ct.transform(row)  # row is a DF with exactly one row

        # 2) predict
        pred = int(xgb_model.predict(X_numeric)[0])
        st.metric('Prediction', 'Satisfied' if pred == 1 else 'Unsatisfied')
        
        inputs = row.drop(columns=['satisfaction_target'], errors='ignore').iloc[0].to_dict()
        
        st.session_state.last_inputs = inputs
        st.session_state.last_pred = pred

        reply = generate_response(inputs, pred)
        st.header('AI Reply')
        st.write(reply)
        
        # Auto email
        if customer_email:
            creds = st.secrets['email']
            with smtplib.SMTP_SSL(creds['smtp_server'], creds['port']) as smtp:
                smtp.login(creds['user'], creds['password'])
                subject = 'Regarding your Flipkart support experience'
                msg = f"Subject: {subject}\n\n{reply}".encode('utf-8')
                smtp.sendmail(creds['user'], customer_email, msg)
            st.success(f"Email sent to {customer_email}")
        else:
            st.error('Enter customer email to send reply.')

    # 4) Chatbot section - Manager only
    if is_manager:
        st.markdown("---")
        st.header("Manager Assistant Chatbot")

        if 'last_inputs' in st.session_state and 'last_pred' in st.session_state:
            st.subheader("Last Case Details")
            st.json(st.session_state.last_inputs)
            st.metric("Prediction", "Satisfied" if st.session_state.last_pred == 1 else "Unsatisfied")
        else:
            st.info("No prediction available yet. Please run a prediction.")

        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state.history = []

        # Show chat history
        for speaker, text in st.session_state.history:
            with st.chat_message(speaker):
                st.write(text)

        # Chat input for manager
        user_msg = st.chat_input("Ask about agent/supervisor performance, improvements, etc...")
        if user_msg:
            st.session_state.history.append(('user', user_msg))

            try:
                if 'last_inputs' in st.session_state and 'last_pred' in st.session_state:
                    manager_prompt = (
                        "You are a smart assistant helping a Flipkart manager analyze customer support issues.\n"
                        "You will give internal analysis, agent/supervisor evaluation, and suggestions for improvement.\n\n"
                        f"Case Details:\n"
                        f"Category: {st.session_state.last_inputs.get('category')} | Sub-category: {st.session_state.last_inputs.get('Sub-category')}\n"
                        f"City: {st.session_state.last_inputs.get('Customer_City')} \n"
                        f"Experience Level: {st.session_state.last_inputs.get('Experience_Level')} | "
                        f"Remarks: {st.session_state.last_inputs.get('Customer Remarks')}\n"
                        f"Handled by: Agent: {st.session_state.last_inputs.get('Agent_name')}, Supervisor: {st.session_state.last_inputs.get('Supervisor')}, Manager: {st.session_state.last_inputs.get('Manager')}\n"
                        f"Prediction: {'Satisfied' if st.session_state.last_pred == 1 else 'Unsatisfied'}\n"
                        f"Manager Question: {user_msg}\n\n"
                        "Assistant Reply:"
                    )
                    chat = genai_model.start_chat()
                    resp = chat.send_message(manager_prompt)
                    assistant_msg = resp.text.strip()
                else:
                    assistant_msg = "⚠️ Please run a prediction first."

            except Exception as e:
                assistant_msg = "❌ Error while generating response."
                st.error(f"Error: {e}")

            st.session_state.history.append(('assistant', assistant_msg))
            with st.chat_message("assistant"):
                st.write(assistant_msg)



if __name__ == '__main__':
    main()
