"""Demo: Discover process-quality relationships in manufacturing."""

import numpy as np
import pandas as pd
from eureka.regression import discover
from eureka.simplify import simplify_tree


def demo_injection_molding():
    """defect_rate ~ 0.01*(temp-230)^2 + 0.5*exp(speed/30)"""
    print("=== Injection Molding: Defect Rate Prediction ===")
    np.random.seed(42)
    n = 200

    temp = np.random.uniform(180, 280, n).astype(np.float32)
    pressure = np.random.uniform(50, 150, n).astype(np.float32)
    speed = np.random.uniform(10, 60, n).astype(np.float32)

    defect = (0.01 * (temp - 230)**2 + 0.5 * np.exp(speed / 30)
              + np.random.randn(n).astype(np.float32) * 2).astype(np.float32)
    defect = np.clip(defect, 0, None)

    df = pd.DataFrame({
        "temperature": temp, "pressure": pressure,
        "speed": speed, "defect_rate": defect,
    })
    csv_path = "/Users/laptopjuan/Documents/eureka-eml/demos/manufacturing_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print("Try: eureka demos/manufacturing_data.csv --target defect_rate\n")

    X = np.column_stack([temp, pressure, speed])
    result = discover(X, defect, population=30, generations=40, timeout=90, verbose=True)

    print(f"\nDiscovered: defect_rate = {simplify_tree(result.tree)}")
    print(f"True: ~ 0.01*(temp-230)^2 + 0.5*exp(speed/30)")
    print(f"MSE: {result.mse:.6e}")


if __name__ == "__main__":
    print("Eureka-EML Manufacturing Demo")
    print("=" * 50 + "\n")
    demo_injection_molding()
