import streamlit as st
from joblib import load
from time import time
import pandas as pd
import plotly.express as px

# %%


@st.cache_resource
def load_model():
    return load("char_text_clf_model.joblib")


# %%

model = load_model()

st.set_page_config(
    page_title="Text Classifier",
    page_icon="🔤",
    layout="centered")

st.title("Real-Time Text Classifier")

user_input = st.text_input("Text here", placeholder="Start typing...")


def draw_bars(label_probs: dict):

    dt = pd.Series(label_probs).sort_index(ascending=False)

    fig = px.bar(
        dt,
        x=dt.values,
        y=dt.index,
        orientation='h',
        text=dt.values,
        # color=dt.index,
        # color_discrete_sequence=px.colors.qualitative.Plotly,
        range_x=[0, 1.1],
        height=250
    )

    fig.update_traces(
        texttemplate='%{text:.1%}',
        textposition='outside'
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=5, r=5, t=5, b=5),
        bargap=0.3,
        xaxis=dict(
            tickformat='.0%',
            title=None,
            # showgrid=True,
            # gridcolor='lightgrey'
        ),
        yaxis=dict(
            title=None,
            showgrid=False
        )
    )

    return fig


if user_input.strip():
    start = time()
    probs = model.predict(user_input)
    elapsed = (time() - start) * 1000  # ms

    figure = draw_bars(probs)
    st.plotly_chart(figure, use_container_width=True)
    st.caption(f"⏱️ {elapsed:.1f} ms")
