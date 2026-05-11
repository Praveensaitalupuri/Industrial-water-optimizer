import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Water Treatment Optimizer", layout="wide")
st.title("🏭 Industrial Wastewater Treatment Optimization Dashboard")

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    columns = ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SV-E', 'SED-E', 'COND-E',
               'PH-P', 'DBO-P', 'SS-P', 'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D',
               'SS-D', 'SV-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 'DQO-S', 'SS-S',
               'SV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-DQO-P', 'RD-SS-P', 'RD-SED-P',
               'RD-DBO-S', 'RD-DQO-S', 'RD-SS-S', 'RD-SED-S', 'RD-DBO-G', 'RD-DQO-G',
               'RD-SS-G', 'RD-SED-G']
    
    df = pd.read_csv('water-treatment.data', sep=',', names=columns, na_values='?')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())
    return df

df = load_data()

# ====================== LOAD MODEL ======================
model = joblib.load('water_model.pkl')

# ====================== SIDEBAR ======================
st.sidebar.header("🔧 Adjust Input Parameters")

# Use only the features the model was trained on
input_features = ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SV-E', 
                  'SED-E', 'COND-E', 'PH-P', 'DBO-P', 'SS-P', 'PH-D', 'SS-D']

inputs = {}
for col in input_features:
    inputs[col] = st.sidebar.slider(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Prediction", "📈 Insights"])

with tab1:
    st.subheader("Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Output SS-S", f"{df['SS-S'].mean():.2f}")
    c2.metric("Avg Output pH", f"{df['PH-S'].mean():.2f}")
    c3.metric("Global Performance", f"{df['RD-SS-G'].mean():.2f}")

    fig = px.scatter(df, x='SS-E', y='SS-S', color='PH-E', 
                     title="Input Solids vs Output Solids")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Real-time Prediction")
    input_df = pd.DataFrame([inputs])
    pred = model.predict(input_df)[0]
    
    st.success(f"**Predicted Output SS-S: {pred:.2f}**")
    
    if pred > 40:
        st.error("⚠️ HIGH SOLIDS - Optimization Recommended")
    else:
        st.success("✅ Good Effluent Quality")

with tab3:
    st.subheader("Feature Importance")
    imp = pd.Series(model.feature_importances_, index=input_features).sort_values(ascending=False)
    st.bar_chart(imp)

st.caption("✅ Fixed Version | Industrial Process Optimization Dashboard")