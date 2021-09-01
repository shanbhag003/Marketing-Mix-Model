import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pulp
from pulp import *


#Sets the layout to full width
st.set_page_config(layout= "wide")


#Web App Title
st.title('''
**Marketing Mix Model**''')

# Upload data (CSV File Only)
with st.sidebar.header('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

#---------------------------------------------------------#

#Sales Optimzation Model Building
def build_sales_model(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    a = lr.coef_
    Coefficients = {"TV Coefficient": a[0], "Radio Coefficient": a[1], "Newspaper Coefficient": a[2]}

    #Optimization (Initializing the model)
    sales_model = LpProblem("Maximize_Sales", sense=LpMaximize)

    #Defining the decision variables
    TV = LpVariable("TV", lowBound=0, upBound=None, cat='Continuous')
    radio = LpVariable("radio", lowBound=0, upBound=None, cat='Continuous')
    newspaper = LpVariable("newspaper", lowBound=0, upBound=None, cat='Continuous')

    #Defining the objective function
    sales_model += Coefficients["TV Coefficient"] * TV + Coefficients["Radio Coefficient"] * radio + Coefficients[
        "Newspaper Coefficient"] * newspaper

    #Defining objective constraints
    sales_model += TV <= parameter_TV_constraint
    sales_model += radio <= parameter_radio_constraint
    sales_model += newspaper <= parameter_newspaper_constraint
    sales_model += TV + radio + newspaper <= parameter_total_budget_constraint

    #Solving method
    sales_model.solve()

    #Status of the solution (Optimal/Not Optimal)
    st.write('Status of the Sales Model (Optimal/Not Optimal) ')
    st.info(LpStatus[sales_model.status])

    #Optimized variables
    st.write("Optimized TV Budget Allocation: ")
    st.info(TV.varValue)
    Optimized_TV_budget = TV.varValue
    st.write("Optimized Radio Budget Allocation: ")
    st.info(radio.varValue)
    Optimized_radio_budget = radio.varValue
    st.write("Optimized Newspaper Budget Allocation: ")
    st.info(newspaper.varValue)
    Optimized_newspaper_budget = newspaper.varValue

    #Optimized final value
    st.write("Optimized Sales: ")
    Total_optimized_sales = value(sales_model.objective)
    st.info(value(sales_model.objective))

    return [Optimized_TV_budget,Optimized_radio_budget,Optimized_newspaper_budget]

#---------------------------------------------------------#

#Units Optimization Model Building

def build_units_model(Optimized_TV_budget,Optimized_radio_budget,Optimized_newspaper_budget):

    #Optimization (Initializing the model)
    units_model = LpProblem("Maximize_Sales", sense=LpMaximize)

    #Defining the decision variables
    TV_units = LpVariable("TV_units", lowBound=0, upBound=None, cat='Integer')
    radio_units = LpVariable("radio_units", lowBound=0, upBound=None, cat='Integer')
    newspaper_units = LpVariable("newspaper_units", lowBound=0, upBound=None, cat='Integer')

    #Defining the objective function
    units_model +=  parameter_TV_units_constraint * TV_units + parameter_radio_units_constraint * radio_units + \
                    parameter_newspaper_units_constraint * newspaper_units

    #Defining objective constraints
    units_model += TV_units <= parameter_TV_units_constraint
    units_model += radio_units <= parameter_radio_units_constraint
    units_model += newspaper_units <= parameter_newspaper_units_constraint
    units_model += parameter_TV_units_constraint * TV_units <= Optimized_TV_budget
    units_model += parameter_radio_units_constraint * radio_units <= Optimized_radio_budget
    units_model += parameter_newspaper_units_constraint * newspaper_units <= Optimized_newspaper_budget

    #Solving method
    units_model.solve()

    #Status of the solution (Optimal/Not Optimal)
    st.write('Status of the Units Model (Optimal/Not Optimal) ')
    st.info(LpStatus[units_model.status])

    #Optimized variables
    st.write("Total TV ad units that can be bought: ")
    st.info(TV_units.varValue)
    Optimized_TV_ad_units = TV_units.varValue
    st.write("Total radio ad units that can be bought: ")
    st.info(radio_units.varValue)
    Optimized_radio_ad_units = radio_units.varValue
    st.write("Total newspaper ad units that can be bought: ")
    st.info(newspaper_units.varValue)
    Optimized_newspaper_ad_units = newspaper_units.varValue

    #Optimized final value
    st.write("Total Advertisement units: ")
    st.info(Optimized_TV_ad_units + Optimized_radio_ad_units + Optimized_newspaper_ad_units)


#---------------------------------------------------------#

#Sidebar - Specify Parameter settings
with st.sidebar.subheader("1. Budgetory Constraints"):
    parameter_TV_constraint = st.sidebar.slider('TV advertising Budget (in thousand rupees)', 0, 1000, 100, 1)
    parameter_radio_constraint = st.sidebar.slider('Radio advertising Budget (in thousand rupees)', 0, 1000, 100, 1)
    parameter_newspaper_constraint = st.sidebar.slider('Newspaper advertising Budget (in thousand rupees)', 0, 1000, 100, 1)
    parameter_total_budget_constraint = st.sidebar.slider('Total advertising Budget (in thousand rupees)', 0, 1000, 100, 1)

with st.sidebar.subheader("2. Quantity Constraints"):
    parameter_TV_units_constraint = st.sidebar.slider('Per unit cost of TV ad (in thousand rupees)', 0, 1000, 100, 1)
    parameter_radio_units_constraint = st.sidebar.slider('Per unit cost of radio ad (in thousand rupees)', 0, 1000, 100, 1)
    parameter_newspaper_units_constraint = st.sidebar.slider('Per unit cost of newspaper ad (in thousand rupees)', 0, 1000, 100, 1)

#---------------------------------------------------------#


#Main Panel

st.subheader("Dataset")

global df
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**Glimpse of Dataset**')
    st.write(df)
    st.header('***Sales Optimization Model***')
    Optimized_TV_budget, Optimized_radio_budget, Optimized_newspaper_budget = build_sales_model(df)
    st.write('---')
    st.header('***Advertisement Units Model***')
    build_units_model(Optimized_TV_budget, Optimized_radio_budget, Optimized_newspaper_budget)

else:
    st.info("Awaiting for CSV file to be uploaded.")
