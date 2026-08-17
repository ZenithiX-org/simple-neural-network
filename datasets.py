
import numpy as np

def get_dataset(name):
    if name in ("XOR", "AND", "OR"):
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        targets = {
            "XOR": [0,1,1,0],
            "AND": [0,0,0,1],
            "OR": [0,1,1,1]
        }
        return X, np.array(targets[name], dtype=float).reshape(-1,1)

    rng = np.random.default_rng(7)
    n = 500

    if name == "Circle":
        X = rng.uniform(-1.5, 1.5, (n,2))
        y = ((X[:,0]**2 + X[:,1]**2) < .7**2).astype(float)
        return X, y.reshape(-1,1)

    if name == "Moons":
        m = n//2
        a = rng.uniform(0, np.pi, m)
        X1 = np.c_[np.cos(a), np.sin(a)]
        b = rng.uniform(0, np.pi, m)
        X2 = np.c_[1-np.cos(b), .5-np.sin(b)]
        X = np.vstack([X1,X2]) + rng.normal(0,.08,(n,2))
        y = np.r_[np.zeros(m), np.ones(m)]
        return X, y.reshape(-1,1)

    raise ValueError("Unknown dataset")
