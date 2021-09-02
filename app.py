import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pulp import *
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


#Sets the layout to full width
st.set_page_config(layout= "wide")

image = Image.open("logo.png")
st.image(image)

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
    c = lr.intercept_
    Coefficients = {"TV Coefficient": a[0], "Radio Coefficient": a[1], "Newspaper Coefficient": a[2]}


    #Optimization (Initializing the model)
    sales_model = LpProblem("Maximize_Sales", sense=LpMaximize)


    #Defining the decision variables
    TV = LpVariable("TV", lowBound=0, upBound=None, cat='Integer')
    radio = LpVariable("radio", lowBound=0, upBound=None, cat='Integer')
    newspaper = LpVariable("newspaper", lowBound=0, upBound=None, cat='Integer')


    #Defining the objective function
    sales_model += Coefficients["TV Coefficient"] * TV + Coefficients["Radio Coefficient"] * radio + Coefficients[
        "Newspaper Coefficient"] * newspaper + c


    #Defining objective constraints
    sales_model += parameter_TV_units_constraint * TV <= parameter_TV_constraint
    sales_model += parameter_radio_units_constraint * radio <= parameter_radio_constraint
    sales_model += parameter_newspaper_units_constraint * newspaper <= parameter_newspaper_constraint
    sales_model += parameter_TV_units_constraint * TV + parameter_radio_units_constraint * radio + \
                   parameter_newspaper_units_constraint * newspaper <= parameter_total_budget_constraint


    #Solving method
    sales_model.solve()


    #Status of the solution (Optimal/Not Optimal)
    st.write('Status of the Sales Model (Optimal/Not Optimal) ')
    st.info(LpStatus[sales_model.status])


    #Optimum Budget for Mediums
    st.write("Optimized TV Budget Allocation: ")
    st.info(TV.varValue*parameter_TV_units_constraint)

    st.write("Optimized Radio Budget Allocation: ")
    st.info(radio.varValue*parameter_radio_units_constraint)

    st.write("Optimized Newspaper Budget Allocation: ")
    st.info(newspaper.varValue*parameter_newspaper_units_constraint)


    #Optimized final sales value
    st.write("Optimized Sales: ")
    st.info(value(sales_model.objective))


    #Optimized variables
    st.write("Total TV ad units that can be bought: ")
    st.info(TV.varValue)

    st.write("Total Radio ad units that can be bought: ")
    st.info(radio.varValue)

    st.write("Total Newspaper ad units that can be bought: ")
    st.info(newspaper.varValue)


    #Optimized final cost value
    st.write("Total Cost: ")
    st.info(TV.varValue*parameter_TV_units_constraint + radio.varValue*parameter_radio_units_constraint +
            newspaper.varValue*parameter_newspaper_units_constraint)

    return [X,lr]


#---------------------------------------------------------#

#Sidebar - Specify Parameter settings
with st.sidebar.subheader("Input Values"):
    parameter_TV_constraint = st.sidebar.slider('What is your TV advertisement budget? (in thousand rupees)', 0, 1000, 100, 1)
    parameter_radio_constraint = st.sidebar.slider('What is your Radio advertisement budget? (in thousand rupees)', 0, 1000, 100, 1)
    parameter_newspaper_constraint = st.sidebar.slider('What is your Newspaper advertisement budget? (in thousand rupees)', 0, 1000, 100, 1)
    parameter_total_budget_constraint = st.sidebar.slider('What is your Total advertisement budget? (in thousand rupees)', 0, 1000, 200, 1)
    parameter_TV_units_constraint = st.sidebar.slider('How much does per unit TV ad cost? (in thousand rupees)', 0, 1000, 20, 1)
    parameter_radio_units_constraint = st.sidebar.slider('How much does per unit Radio ad cost? (in thousand rupees)', 0, 1000, 5, 1)
    parameter_newspaper_units_constraint = st.sidebar.slider('How much does per unit Newspaper ad cost? (in thousand rupees)', 0, 1000, 10, 1)

#---------------------------------------------------------#

#Main Panel

st.subheader("Dataset")


global df
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.header('**Glimpse of Dataset**')
    st.write(df)
    st.header('***Sales Optimization Model***')
    X,lr = build_sales_model(df)
    data = [lr.coef_]
    df1 = pd.DataFrame(data,columns = X.columns)
    #Plotting coefficients to check the behaviour of each medium
    st.header('**Influence of Advertisement Mediums on Sales**')
    st.bar_chart(df1)

else:
    st.info("Awaiting for CSV file to be uploaded.")

#-------------------------------------------------------------#
