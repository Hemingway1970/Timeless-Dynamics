import numpy as np
import itertools

def simulate_timeless_dynamics(delta=2.0):
    """
    Implementation of the Record-Bias Toy Model (Lombardo, 2026).
    Demonstrates emergent ordering in a timeless configuration space.
    """
    # 1. Configuration Space (A=System, B=Environment, M=Record)
    configs = list(itertools.product([0, 1], repeat=3))
    
    print(f"--- Timeless Dynamics Audit: Delta={delta} ---")
    
    # 2. Potential Function V(c) - The "Friction"
    # Favors states where the 'Record' M matches the 'System' A.
    weights = []
    for c in configs:
        A, B, M = c
        potential = 0 if M == A else delta
        weight = np.exp(-potential)
        weights.append(weight)
    
    # 3. Probability Distribution (The 'Static' Ensemble)
    probs = np.array(weights) / sum(weights)
    
    # 4. Results Audit
    print(f"{'State (A,B,M)':<15} | {'Probability':<12} | {'Status'}")
    print("-" * 45)
    for i, c in enumerate(configs):
        status = "FAVORED (Record Match)" if c[0] == c[2] else "PENALIZED"
        print(f"{str(c):<15} | {probs[i]:.4f}      | {status}")

if __name__ == "__main__":
    # Test the model with high bias to see the 'Arrow' emerge
    simulate_timeless_dynamics(delta=2.5)