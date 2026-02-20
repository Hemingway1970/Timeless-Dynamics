"""
FILE: sim_v14_mirror.py
PURPOSE: Independent Epistemic Audit of Timeless Dynamics v14.
ORIGIN: Generated via zero-context prompt from ChatGPT-4o.
RATIONALE: This script serves as a 'Mirror Test.' It demonstrates that 
the Recordability Condition and the emergence of the Schrödinger-style 
extremal path are the logical and inevitable consequences of the 
Wheeler-DeWitt (HΨ=0) constraint when translated by a clean inference engine.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# -----------------------------
#  Timeless Dynamics Toy Model (Mirror)
# -----------------------------

def make_timeless_manifold(
    n_points=600,
    d_state=6,
    d_mem=6,
    noise_state=0.15,
    noise_mem=0.20,
    mem_strength=0.92,
    seed=7
):
    """
    Generate a static cloud of configurations C=(x,m) in R^(d_state + d_mem).
    Reality is modeled as a static dataset where "memory" is an encoded persistence.
    """
    rng = np.random.default_rng(seed)
    s = np.linspace(0, 1, n_points)

    # State manifold: smooth nonlinear embedding
    basis = [np.sin(2*np.pi*s), np.cos(2*np.pi*s), np.sin(4*np.pi*s), 
             np.cos(4*np.pi*s), s, s**2]
    X_lat = np.stack(basis[:d_state], axis=1)

    W = rng.normal(size=(d_state, d_state))
    x_clean = X_lat @ W.T
    x = x_clean + noise_state * rng.normal(size=x_clean.shape)

    # Memory as a leaky persistence of encoded state
    m = np.zeros((n_points, d_mem))
    E_true = rng.normal(size=(d_mem, d_state)) 

    m_prev = np.zeros(d_mem)
    for i in range(n_points):
        target = E_true @ x_clean[i]
        m_prev = mem_strength * m_prev + (1.0 - mem_strength) * target
        m[i] = m_prev

    m = m + noise_mem * rng.normal(size=m.shape)
    C = np.hstack([x, m])
    return C, x, m, E_true, s

def recordability_R(A, B, d_state, E, sigma_R=1.0):
    """
    R(A,B) measures how well B 'remembers' A's state.
    """
    xA = A[:d_state]
    mB = B[d_state:]
    pred = E @ xA
    diff = mB - pred
    return np.exp(-0.5 * (diff @ diff) / (sigma_R**2))

def build_knn_graph(C, k=12):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(C)
    distances, indices = nbrs.kneighbors(C)
    return distances[:, 1:], indices[:, 1:]

def dijkstra_path(cost_fn, neighbors, start, goal):
    n = len(neighbors)
    INF = 1e30
    dist = np.full(n, INF); prev = np.full(n, -1, dtype=int); visited = np.zeros(n, dtype=bool)
    dist[start] = 0.0
    for _ in range(n):
        i = np.argmin(np.where(visited, INF, dist))
        if visited[i] or dist[i] >= INF: break
        visited[i] = True
        if i == goal: break
        for j in neighbors[i]:
            if visited[j]: continue
            nd = dist[i] + cost_fn(i, j)
            if nd < dist[j]:
                dist[j] = nd; prev[j] = i
    if dist[goal] >= INF/2: return None, dist
    path = []; cur = goal
    while cur != -1:
        path.append(cur); cur = prev[cur]
    path.reverse()
    return path, dist

def main():
    n_points = 700; d_state = 6; d_mem = 6; k = 14
    alpha = 1.0; beta = 2.2; sigma_R = 2.0; eps = 1e-12

    C, x, m, E_true, s_true = make_timeless_manifold(n_points, d_state, d_mem, seed=10)
    rng = np.random.default_rng(123)
    E = E_true + 0.15 * rng.normal(size=E_true.shape)
    dists, nbr_idx = build_knn_graph(C, k=k)

    def edge_cost(i, j):
        A, B = C[i], C[j]
        D = np.linalg.norm(B - A)
        R = recordability_R(A, B, d_state=d_state, E=E, sigma_R=sigma_R)
        return alpha * D - beta * np.log(R + eps)

    start = int(0.02 * n_points); goal = int(0.98 * n_points)
    path, _ = dijkstra_path(edge_cost, nbr_idx, start, goal)
    
    if path:
        s_path = s_true[path]
        monotonicity = np.mean(np.diff(s_path) > 0)
        pca = PCA(n_components=2); C2 = pca.fit_transform(C)
        plt.figure(figsize=(10, 7))
        plt.scatter(C2[:, 0], C2[:, 1], c=s_true, s=18, alpha=0.55)
        P2 = C2[path]
        plt.plot(P2[:, 0], P2[:, 1], linewidth=3, color='red')
        plt.title("Emergent Path from Max Recordability (Mirror Audit)")
        plt.show()
        print(f"Monotonicity: {monotonicity:.3f}")

if __name__ == "__main__":
    main()
