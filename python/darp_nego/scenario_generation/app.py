import streamlit as st

# To run: streamlit run app.py

st.set_page_config(page_title="DARP App", page_icon="🧠")

st.title("Welcome to the DARP App")
st.markdown("""
Choose a task from the sidebar:

- 🛠️ Generate DARP cases
- 🤖 Solve generated cases
""")
