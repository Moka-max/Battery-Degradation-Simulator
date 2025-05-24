import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def generate_capacity(cycles_arr, battery_type, temp, rate):
    base = {"Li-ion": 100, "Solid State": 105, "LFP": 95}[battery_type]
    decay_factor = {"Li-ion": 0.0002, "Solid State": 0.0001, "LFP": 0.00015}[battery_type]
    temp_factor = (1 + abs(temp - 25) * 0.005)
    rate_factor = (1 + (rate - 1) * 0.04)
    noise = np.random.normal(0, 1.5, size=cycles_arr.shape)
    return (base * np.exp(-decay_factor * cycles_arr.flatten()) / (temp_factor * rate_factor)) + noise.flatten()

def main():
    print("=== Battery Capacity Degradation Simulator ===\n")
    
    battery_types = ["Li-ion", "Solid State", "LFP"]
    for i, btype in enumerate(battery_types, 1):
        print(f"{i}. {btype}")
    b_choice = int(input("Select Battery Type (1-3): "))
    battery_type = battery_types[b_choice - 1]

    temp = float(input("Enter Operating Temperature (°C) [0-60]: "))
    rate = float(input("Enter Charging Rate (C-rate) [1-3]: "))
    cycles = int(input("Enter Number of Charge Cycles [0-3000]: "))

    x = np.arange(0, 3001, 300).reshape(-1, 1)
    y = generate_capacity(x, battery_type, temp, rate)

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(X_poly, y)

    input_cycle = np.array([[cycles]])
    input_poly = poly.transform(input_cycle)
    predicted_capacity = model.predict(input_poly)[0]

    print(f"\nPredicted Battery Capacity after {cycles} cycles: {predicted_capacity:.2f}%")

    plt.scatter(x, y, color='blue', label='Simulated Data')
    plt.plot(x, model.predict(poly.transform(x)), color='red', label='Polynomial Fit')
    plt.scatter(cycles, predicted_capacity, color='green', s=100, label='Prediction')
    plt.xlabel("Charge Cycles")
    plt.ylabel("Capacity (%)")
    plt.ylim(0, 110)
    plt.legend()
    plt.title("Battery Capacity Degradation Curve")
    plt.show()

if __name__ == "__main__":
    main()
