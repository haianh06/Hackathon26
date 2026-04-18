import streamlit as st

st.write("Main App")

@st.fragment(run_every=1.0)
def auto_frag():
    st.write("Auto frag")
    if st.button("Refresh", key="refresh"):
        st.write("CLICKED")
        st.rerun()

auto_frag()
