import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.title("Предсказание стоимости автомобилей")

# Загрузка модели и данных из pickle
@st.cache_resource
def load_pickle():
    with open('model.pickle', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler'], model_data['feature_columns'], model_data['df_train']

model, scaler, feature_columns, df_train = load_pickle()

st.header("Ключевые графики EDA")

tab1, tab2, tab3 = st.tabs(["Распределения", "Корреляция", "Зависимости"])

with tab1:
    st.subheader("Распределение числовых признаков")
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for i, col in enumerate(numeric_cols[:6]):
        ax = axes[i // 3, i % 3]
        ax.hist(df_train[col].dropna(), bins=30, edgecolor='black')
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Матрица корреляций")
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df_train.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    st.pyplot(fig)

with tab3:
    st.subheader("Зависимость цены от признаков")
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'selling_price']
    selected_feature = st.selectbox("Выберите признак:", numeric_cols)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_train[selected_feature], df_train['selling_price'], alpha=0.3)
    ax.set_xlabel(selected_feature)
    ax.set_ylabel('selling_price')
    st.pyplot(fig)

st.header("Предсказание стоимости")

input_method = st.radio("Способ ввода данных:", ["Ввод вручную", "Загрузка CSV"])

if input_method == "Ввод вручную":
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Год выпуска", min_value=1980, max_value=2025, value=2015)
        km_driven = st.number_input("Пробег (км)", min_value=0, max_value=1000000, value=50000)
        mileage = st.number_input("Расход топлива (км/л)", min_value=0.0, max_value=50.0, value=18.0)
        engine = st.number_input("Объем двигателя (куб.см)", min_value=500, max_value=6000, value=1200)
    
    with col2:
        max_power = st.number_input("Макс. мощность (л.с.)", min_value=30.0, max_value=500.0, value=80.0)
        torque = st.number_input("Крутящий момент (Н * м)", min_value=50.0, max_value=800.0, value=150.0)
        seats = st.number_input("Количество мест", min_value=2, max_value=10, value=5)
        max_torque_rpm = st.number_input("Обороты макс. момента", min_value=1000.0, max_value=8000.0, value=3500.0)
    
    if st.button("Предсказать цену"):
        input_data = pd.DataFrame([[year, km_driven, mileage, engine, max_power, torque, seats, max_torque_rpm]], 
                                   columns=feature_columns)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Предсказанная стоимость: {prediction:,.0f} у. е.")
else:
    uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Загруженные данные:")
        st.dataframe(input_df.head())
        
        if st.button("Предсказать цены"):
            input_data = input_df[feature_columns]
            input_scaled = scaler.transform(input_data)
            predictions = model.predict(input_scaled)
            
            result_df = input_df.copy()
            result_df['predicted_price'] = predictions
            st.write("Предсказанные цены:")
            st.dataframe(result_df)

st.header("Веса модели")

weights_df = pd.DataFrame({
    'Признак': feature_columns,
    'Вес': model.coef_
}).sort_values('Вес', key=abs, ascending=False) # отсортируем по абсолютному значению веса

st.dataframe(weights_df)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['green' if w > 0 else 'red' for w in weights_df['Вес']]
ax.barh(weights_df['Признак'], weights_df['Вес'], color=colors)
ax.set_xlabel('Вес')
ax.set_title('Важность признаков (веса модели)')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
st.pyplot(fig)

st.info("Зелёные полосы — положительное влияние на цену, красные — отрицательное")
