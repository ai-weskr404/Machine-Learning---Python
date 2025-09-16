import numpy as np
import matplotlib.pyplot as plt
     
# Helper functions for linear regression calculations

def summation(arr: list[float]) -> float:
    """Calculate the sum of all elements in an array"""
    total = 0
    for i in arr:
        total += i
    return total

def square(arr: list[float]) -> float:
    """Calculate the sum of squared elements in an array"""
    return np.sum(np.square(arr))

def pair(x: list[float], y: list[float]) -> float:
    """Calculate the sum of products of corresponding elements in two arrays"""
    total = 0
    for i in range(len(x)):
        total += x[i] * y[i]
    return total

def a(x: list[float], y: list[float]) -> float:
    """Calculate the y-intercept (a) of the linear regression line"""
    n = len(x)
    denominator = n * square(x) - summation(x) ** 2
    if denominator == 0:
        raise ValueError("DENOMINATOR IN CALCULATION OF 'A' IS ZERO. ALL X VALUES MAY BE THE SAME.")
    return (summation(y) * square(x) - summation(x) * pair(x, y)) / denominator

def b(x: list[float], y: list[float]) -> float:
    """Calculate the slope (b) of the linear regression line"""
    n = len(x)
    denominator = n * square(x) - summation(x) ** 2
    if denominator == 0:
        raise ValueError("DENOMINATOR IN CALCULATION OF 'B' IS ZERO. ALL X VALUES MAY BE THE SAME.")
    return (n * pair(x, y) - summation(x) * summation(y)) / denominator

def linear_regression(x: list[float], y: list[float], value: float) -> float:
    """Predict y value for a given x value using the linear regression equation y = a + bx"""
    return a(x, y) + b(x, y) * value

def r_squared(x: list[float], y: list[float]) -> float:
    """Calculate the coefficient of determination (R^2)"""
    y_mean = np.mean(y)
    y_pred = [linear_regression(x, y, xi) for xi in x]
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - y_hat) ** 2 for yi, y_hat in zip(y, y_pred))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - ss_res / ss_tot

def save_results_to_file(filename: str, x, y, a_val, b_val, r2_val):
    with open(filename, 'w') as f:
        f.write("LINEAR REGRESSION RESULTS\n".upper())
        f.write(f"X = {x}\n".upper())
        f.write(f"Y = {y}\n".upper())
        f.write(f"A (INTERCEPT) = {a_val}\n")
        f.write(f"B (SLOPE) = {b_val}\n")
        f.write(f"R**2 = {r2_val}\n".upper())
        f.write(f"REGRESSION EQUATION: Y = {a_val} + {b_val} X\n".upper())

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt.upper()))
        except ValueError:
            print("INVALID INPUT. PLEASE ENTER A VALID NUMBER.")

def get_int_input(prompt):
    while True:
        try:
            val = int(input(prompt.upper()))
            if val <= 0:
                print("PLEASE ENTER A POSITIVE INTEGER.")
                continue
            return val
        except ValueError:
            print("INVALID INPUT. PLEASE ENTER A VALID INTEGER.")

import sys
import os


def main():
    print("LINEAR REGRESSION".upper())
    x = []
    y = []

    # Input data points with validation
    n = get_int_input("\nENTER THE NUMBER OF DATA POINTS: ")
    for i in range(n):
        while True:
            try:
                vals = input(f"ENTER THE VALUE OF X AND Y (SPACE SEPARATED) FOR POINT {i+1}: ".upper()).split()
                if len(vals) != 2:
                    print("PLEASE ENTER EXACTLY TWO NUMBERS.")
                    continue
                xi, yi = float(vals[0]), float(vals[1])
                x.append(xi)
                y.append(yi)
                break
            except ValueError:
                print("INVALID INPUT. PLEASE ENTER TWO VALID NUMBERS.")

    # Display input data and calculated values
    print()
    print(f"X = {x}".upper())
    print(f"Y = {y}".upper())

    print()
    print(f"X**2 = {[xi**2 for xi in x]}".upper())
    print(f"XY = {[x[i]*y[i] for i in range(len(x))]}".upper())

    print()
    print(f"SUMMATION OF X = {summation(x)}".upper())
    print(f"SUMMATION OF Y = {summation(y)}".upper())

    print()
    print(f"SUMMATION OF X**2 = {square(x)}".upper())
    print(f"SUMMATION OF Y**2 = {square(y)}".upper())

    print()
    print(f"SUMMATION OF XY = {pair(x, y)}".upper())

    # Calculate and display regression coefficients
    try:
        a_val = a(x, y)
        b_val = b(x, y)
    except ValueError as e:
        print(f"ERROR: {e}".upper())
        sys.exit(1)

    print()
    print(f"A = {a_val}".upper())
    print(f"B = {b_val}".upper())

    # Display regression equation
    print()
    print("EQUATION OF LINEAR REGRESSION: ".upper())
    print(f"Y = {a_val} + {b_val} X".upper())

    # Calculate and display R^2
    r2_val = r_squared(x, y)
    print(f"\nR**2 (COEFFICIENT OF DETERMINATION) = {r2_val}".upper())

    # Predict new Y for user-input X
    while True:
        predict_choice = input("\nWOULD YOU LIKE TO PREDICT Y FOR A NEW X VALUE? (Y/N): ".upper()).strip().lower()
        if predict_choice == 'y':
            x_new = get_float_input("ENTER NEW X VALUE: ")
            y_pred = linear_regression(x, y, x_new)
            print(f"PREDICTED Y FOR X = {x_new}: {y_pred}".upper())
        elif predict_choice == 'n':
            break
        else:
            print("PLEASE ENTER 'Y' OR 'N'.")

    # Offer to save results
    save_choice = input("\nWOULD YOU LIKE TO SAVE THE RESULTS TO A TEXT FILE? (Y/N): ".upper()).strip().lower()
    if save_choice == 'y':
        filename = input("ENTER FILENAME (DEFAULT: REGRESSION_RESULTS.TXT): ".upper()).strip()
        if not filename:
            filename = "regression_results.txt"
        save_results_to_file(filename, x, y, a_val, b_val, r2_val)
        print(f"RESULTS SAVED TO {filename}".upper())

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='DATA POINTS')
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(f"({xi},{yi})", (xi, yi), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

    x_line = np.linspace(min(x), max(x), 100)
    y_line = [a_val + b_val * xi for xi in x_line]
    plt.plot(x_line, y_line, color='red', label='REGRESSION LINE')

    plt.xlabel('X'.upper())
    plt.ylabel('Y'.upper())
    plt.title('LINEAR REGRESSION'.upper())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Offer to save plot
    plot_choice = input("\nWOULD YOU LIKE TO SAVE THE PLOT AS AN IMAGE? (Y/N): ".upper()).strip().lower()
    if plot_choice == 'y':
        plot_filename = input("ENTER IMAGE FILENAME (DEFAULT: REGRESSION_PLOT.PNG): ".upper()).strip()
        if not plot_filename:
            plot_filename = "regression_plot.png"
        plt.savefig(plot_filename)
        print(f"PLOT SAVED AS {plot_filename}".upper())

    plt.show()

if __name__ == "__main__":
    main()
