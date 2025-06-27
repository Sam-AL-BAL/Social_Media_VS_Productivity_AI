import streamlit as st
import pandas as pd
import utils

st.title("🔍 Predicción de Productividad")

utils.menu()

# Formulario del usuario
with st.form("formulario"):
    edad = st.number_input("Edad", 18, 100, value=25)
    genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
    trabajo = st.text_input("Trabajo", "Ingeniero")
    tiempo_redes = st.number_input("Horas en redes sociales", 0.0, 24.0, value=3.0)
    red_social = st.text_input("Red social preferida", "Instagram")
    notificaciones = st.number_input("Notificaciones por día", 0, 200, value=50)
    horas_trabajo = st.number_input("Horas de trabajo/día", 0.0, 12.0, value=8.0)
    puntuacion_percibida = st.number_input("Productividad percibida (0-10)", 0.0, 10.0, value=6.0)
    estres = st.number_input("Nivel de estrés (0-10)", 0.0, 10.0, value=4.0)
    sueño = st.number_input("Horas de sueño", 0.0, 12.0, value=7.0)
    pantalla_antes_dormir = st.number_input("Pantalla antes de dormir", 0.0, 5.0, value=1.0)
    descansos = st.number_input("Descansos durante trabajo", 0, 10, value=2)
    usa_apps = st.radio("¿Usa apps de concentración?", ["Sí", "No"])
    bienestar_digital = st.radio("¿Bienestar digital activado?", ["Sí", "No"])
    cafe = st.number_input("Tazas de café al día", 0, 10, value=2)
    burnout = st.number_input("Días con agotamiento al mes", 0, 31, value=5)
    desconexion = st.number_input("Horas de desconexión a la semana", 0.0, 60.0, value=10.0)
    satisfaccion = st.number_input("Satisfacción laboral (0-10)", 0.0, 10.0, value=7.0)

    enviado = st.form_submit_button("Predecir")

if enviado:
    datos_usuario = pd.DataFrame([{
        "age": edad,
        "gender": genero,
        "job_type": trabajo,
        "daily_social_media_time": tiempo_redes,
        "social_platform_preference": red_social,
        "number_of_notifications": notificaciones,
        "work_hours_per_day": horas_trabajo,
        "perceived_productivity_score": puntuacion_percibida,
        "stress_level": estres,
        "sleep_hours": sueño,
        "screen_time_before_sleep": pantalla_antes_dormir,
        "breaks_during_work": descansos,
        "uses_focus_apps": usa_apps,
        "has_digital_wellbeing_enabled": bienestar_digital,
        "coffee_consumption_per_day": cafe,
        "days_feeling_burnout_per_month": burnout,
        "weekly_offline_hours": desconexion,
        "job_satisfaction_score": satisfaccion
    }])

    pred = utils.predecir(datos_usuario)
    st.success(f"📌 Predicción de productividad estimada: {pred:.2f}")


