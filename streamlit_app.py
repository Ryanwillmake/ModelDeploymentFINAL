import streamlit as st
from inference import CreditScoreInferencer

st.set_page_config(page_title="Credit Score Prediction", layout="centered")
st.title("Customer Credit Score Prediction")
st.write("Isi data nasabah di bawah untuk memprediksi kategori credit score (Poor / Standard / Good).")

@st.cache_resource
def load_inferencer():
    return CreditScoreInferencer()

inferencer = load_inferencer()

with st.form("credit_form"):
    month = st.selectbox("Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=35)
        occupation = st.text_input("Occupation", value="Engineer")
        annual_income = st.number_input("Annual Income", min_value=0.0, value=45000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=3500.0)
        num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, value=3)
        num_credit_card = st.number_input("Num Credit Card", min_value=0, value=4)
        interest_rate = st.number_input("Interest Rate", min_value=0.0, value=14.0)
        num_of_loan = st.number_input("Num of Loan", min_value=0.0, value=3.0)
        delay_from_due_date = st.number_input("Delay from Due Date", min_value=0, value=15)
        num_delayed_payment = st.number_input("Num of Delayed Payment", min_value=0.0, value=8.0)

    with col2:
        changed_credit_limit = st.number_input("Changed Credit Limit", value=5.5)
        num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0.0, value=2.0)
        credit_mix = st.selectbox("Credit Mix", ["Standard", "Good", "Bad"])
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1500.0)
        credit_utilization = st.number_input("Credit Utilization Ratio", min_value=0.0, value=30.5)
        credit_history_age = st.text_input("Credit History Age", value="8 Years and 3 Months")
        payment_min_amount = st.selectbox("Payment of Min Amount", ["Yes", "No"])
        total_emi = st.number_input("Total EMI per month", min_value=0.0, value=150.0)
        amount_invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=300.0)
        payment_behaviour = st.selectbox("Payment Behaviour", [
            "High_spent_Small_value_payments", "Low_spent_Small_value_payments",
            "High_spent_Medium_value_payments", "Low_spent_Medium_value_payments",
            "High_spent_Large_value_payments", "Low_spent_Large_value_payments"
        ])
        monthly_balance = st.number_input("Monthly Balance", value=300.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "Month": month,
        "Age": age, "Occupation": occupation, "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary, "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card, "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan, "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": num_delayed_payment, "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_credit_inquiries, "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt, "Credit_Utilization_Ratio": credit_utilization,
        "Credit_History_Age": credit_history_age, "Payment_of_Min_Amount": payment_min_amount,
        "Total_EMI_per_month": total_emi, "Amount_invested_monthly": amount_invested,
        "Payment_Behaviour": payment_behaviour, "Monthly_Balance": monthly_balance
    }

    result = inferencer.predict(input_data)
    color_map = {"Good": "success", "Standard": "warning", "Poor": "error"}
    getattr(st, color_map[result])(f"Predicted Credit Score: **{result}**")
