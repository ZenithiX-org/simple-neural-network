
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from datasets import get_dataset

st.set_page_config(page_title="Neural Network Playground", page_icon="🧠", layout="wide")
st.title("🧠 Neural Network Playground")
st.write("A configurable neural network built from scratch using NumPy.")

st.sidebar.header("⚙️ Settings")
dataset = st.sidebar.selectbox("Dataset", ["XOR","AND","OR","Circle","Moons"])
layers = st.sidebar.slider("Hidden Layers", 1, 3, 2)
neurons = st.sidebar.slider("Neurons per Layer", 2, 32, 8)
lr = st.sidebar.slider("Learning Rate", .001, 1.0, .1, .001)
epochs = st.sidebar.slider("Epochs", 100, 10000, 3000, 100)

X, y = get_dataset(dataset)

st.subheader(f"📊 {dataset} Dataset")
fig, ax = plt.subplots()
ax.scatter(X[y.ravel()==0,0], X[y.ravel()==0,1], label="Class 0")
ax.scatter(X[y.ravel()==1,0], X[y.ravel()==1,1], label="Class 1")
ax.legend()
ax.grid(alpha=.2)
st.pyplot(fig)

st.write("Architecture:", " → ".join(map(str,[2]+[neurons]*layers+[1])))

if st.button("🚀 Train Neural Network", type="primary", use_container_width=True):
    model = NeuralNetwork(2, [neurons]*layers, 1, lr)
    with st.spinner("Training..."):
        losses = model.train(X,y,epochs)

    proba = model.predict_proba(X).ravel()
    pred = (proba >= .5).astype(int)
    acc = np.mean(pred == y.ravel())*100

    a,b,c = st.columns(3)
    a.metric("Accuracy", f"{acc:.1f}%")
    b.metric("Final Loss", f"{losses[-1]:.6f}")
    c.metric("Epochs", f"{epochs:,}")

    st.subheader("📉 Training Loss")
    step=max(1,len(losses)//1000)
    st.line_chart(pd.DataFrame({"Loss":losses[::step]}))

    st.subheader("🎯 Decision Boundary")
    xmin,xmax=X[:,0].min()-.2,X[:,0].max()+.2
    ymin,ymax=X[:,1].min()-.2,X[:,1].max()+.2
    xx,yy=np.meshgrid(np.linspace(xmin,xmax,200),np.linspace(ymin,ymax,200))
    grid=np.c_[xx.ravel(),yy.ravel()]
    zz=model.predict_proba(grid).reshape(xx.shape)
    fig,ax=plt.subplots()
    ax.contourf(xx,yy,zz,levels=30,alpha=.25)
    ax.contour(xx,yy,zz,levels=[.5])
    ax.scatter(X[y.ravel()==0,0],X[y.ravel()==0,1],label="Class 0")
    ax.scatter(X[y.ravel()==1,0],X[y.ravel()==1,1],label="Class 1")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔮 Predictions")
    st.dataframe(pd.DataFrame({
        "Feature 1":X[:,0],
        "Feature 2":X[:,1],
        "Expected":y.ravel().astype(int),
        "Probability":np.round(proba,4),
        "Prediction":pred
    }), use_container_width=True, hide_index=True)
else:
    st.info("Choose settings and click Train Neural Network.")

st.caption("NumPy • ReLU • Sigmoid • Binary Cross Entropy • Gradient Descent")
