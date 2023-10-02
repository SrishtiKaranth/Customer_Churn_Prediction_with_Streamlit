import numpy as np
import pickle 
import streamlit as st
import requests
import json
import streamlit_lottie
from streamlit_lottie import st_lottie

loaded_model = pickle.load(open("trained_model.sav", 'rb'))
loaded_scaler = pickle.load(open("scaler.pkl", 'rb'))

#creating a function 

def churn_prediction(single_obs):
    input_data_as_numpy_array=np.asarray(single_obs)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    standardized_input = loaded_scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(standardized_input)
    if(prediction==1):
       return "Customer lost"
    else:
        return "Customer retained"


def main():

    st.set_page_config(page_title="Customer Churn Prediction",page_icon="🤑",layout="wide")
    with st.container():
        left_column,right_column=st.columns([1,1])
        with left_column:


            st.title("  Cracking the Churn Code: Analytics for Loyalty 🏦")
            st.subheader("Maximizing Customer Lifetime value by Data💸")
            st.markdown(
                        """
                        <div style="text-align: justify;color:grey;">
                        Customer churn prediction is the process of using data analysis and machine learning techniques to identify and forecast customers who are likely to stop using a product or service. By analyzing historical customer behavior, businesses can gain valuable insights into the factors that contribute to customer attrition. This predictive approach allows companies to take proactive measures to retain customers, such as targeted marketing campaigns, personalized incentives, and improved customer service.
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )
            st.write('#')
            st.write("---")
            st.write('#')
        
        with right_column:
            def load_lottiefile(filepath:str):
                with open(filepath,"r") as f:
                    return json.load(f)  # Corrected indentation
            lottie_coding=load_lottiefile("animation_ln90cs9w.json")  # Corrected indentation
            st_lottie(
                lottie_coding,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                # renderer="svg",
                height="400px",
                width="600px",
                key=None,
                )
            st.write('#')  # This line should be outside of the with block


    with st.container():
        st.title("Customer Churn Prediction 🪙")
        st.write("Enter the following details to help us predict if Customer is retained or lost by bank.")
        st.write("##")
        st.write("##")

            #getting input data from user

        creditScore=st.text_input('Enter Credit Score')
        age=st.text_input('Enter Age')
        tenure=st.text_input('Enter Tenure')
        balance=st.text_input('Enter Bank Balance')
        numOfProducts=st.text_input('Enter Number of Products purchased')
        hasCrCard=st.text_input('Does customer have credit card? 1/0(y/n)')
        isActiveMember=st.text_input('Is customer an active member? 1/0(y/n)')
        estimatedSalary=st.text_input('Enter Estimated Salary')
        geography_Germany=st.text_input('Does customer reside in Germany? 1/0(y/n)')
        geography_Spain=st.text_input('Does customer reside in Spain? 1/0(y/n)')
        gender_male=st.text_input('Does customer identify themself as Male? 1/0(y/n)')

        churn=''

        if st.button("Predict Results"):
            churn=churn_prediction([creditScore,age,tenure,balance,numOfProducts,hasCrCard,isActiveMember,estimatedSalary,geography_Germany,geography_Spain,gender_male])

        st.success(churn)
        st.write("This model uses RandomForest and has an accuracy of 87%.")
        st.write("#")
        st.write("---")

    with st.container():
        left_column,right_column=st.columns([1,1])
        with right_column:
            st.subheader("Understanding this project..💰")
            st.markdown(
                        """
                        <div style="text-align: justify;color:grey;">
                        Using a comprehensive dataset from Kaggle, this project explores three powerful machine learning models: Logistic Regression, XGBoost, and Random Forest. These models are tasked with predicting customer churn by analyzing various factors such as credit scores, age, tenure, balance, and more.

                        Among these models, Random Forest emerges as the star performer, boasting the highest accuracy in identifying potential churners. Leveraging its predictive prowess, this model crafts a robust strategy to proactively engage and retain at-risk customers.
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )
            st.write('#')
            st.write("---")
            st.write('#')
        
        with left_column:
            def load_lottiefile(filepath:str):
                with open(filepath,"r") as f:
                    return json.load(f)  # Corrected indentation
            lottie_coding=load_lottiefile("animation_ln91b935.json")  # Corrected indentation
            st_lottie(
                lottie_coding,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                # renderer="svg",
                height="400px",
                width="600px",
                key=None,
                )
            st.write('#')  # This line should be outside of the with block




if __name__ == '__main__':
    main()