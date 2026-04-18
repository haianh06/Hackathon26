import streamlit as st

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
    st.write("I am Tab 1")
    st.components.v1.html("""
    <div id="box" style="width: 100px; height: 100px; background: red;"></div>
    <script>
    setInterval(() => {
        const box = document.getElementById('box');
        console.log("Tab 1 box visibility:", box.offsetWidth > 0);
    }, 1000);
    </script>
    """)

with tab2:
    st.write("I am Tab 2")
