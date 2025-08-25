import numpy as np
import matplotlib.pyplot as plt

from businessnlp.metrics import cosine_similarity


def cosine_similarity_plot_2d(vector, another_vector):
    A = np.array(vector)[:2].reshape(1, 2)
    B = np.array(another_vector)[:2].reshape(1, 2)

    origin = np.array([0, 0])  # origin point

    plt.figure(figsize=(6, 6))
    plt.quiver(
        *origin, *A, angles="xy", scale_units="xy", scale=1, color="r", label="Vector"
    )
    plt.quiver(
        *origin,
        *B,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="b",
        label="Another Vector",
    )

    similarity, radians, degrees = cosine_similarity(vector, another_vector)

    # Annotate angle
    plt.text(1, 1, f"0 = {radians:.2f}", color="purple", fontsize=12)
    plt.text(1, 1, f"degrees = {degrees:.2f}", color="orange", fontsize=12)

    # Set plot limits
    plt.xlim(-1, max(A[0], B[0]) + 1)
    plt.ylim(-1, max(A[1], B[1]) + 1)
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title("Vectors and Angle Between Them")
    plt.show()


def cosine_similarity_plot_3d(vector, another_vector):
    figure = plt.figure(figsize=(7, 7))
    axes = figure.add_subplot(111, projection="3d")

    A = np.array(vector)
    B = np.array(another_vector)
    origin = np.array([0, 0, 0])
    axes.quiver(
        *origin, *A, color="r", label="Vector", linewidth=2, arrow_length_ratio=0.1
    )
    axes.quiver(
        *origin,
        *B,
        color="b",
        label="Another Vector",
        linewidth=2,
        arrow_length_ratio=0.1,
    )

    similarity, radians, degrees = cosine_similarity(vector, another_vector)

    midpoint = (A + B) / 2
    axes.text(*midpoint, f"0 = {radians:.2f}", color="purple", fontsize=12)
    axes.text(*midpoint - 1, f"degrees = {degrees:.2f}", color="orange", fontsize=12)

    max_val = np.max([A, B]) + 1

    axes.set_xlim([0, max_val])
    axes.set_ylim([0, max_val])
    axes.set_zlim([0, max_val])

    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")

    axes.legend()
    axes.set_title("3D Vectors and Angle between them")
    plt.show()


def plot_log(xmax=500):
    x = np.linspace(0.1, 10, xmax)
    y = np.log(x)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="y = log(x)", color="blue", linewidth=2)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Plot of y = log(x)")
    plt.grid(True)
    plt.legend()

    plt.show()
