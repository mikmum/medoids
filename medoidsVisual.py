import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def assign_points_to_medoids(data, medoids):
    clusters = {}
    for point in data:
        distances = [euclidean_distance(point, medoid) for medoid in medoids]
        nearest_medoid_index = np.argmin(distances)
        if nearest_medoid_index in clusters:
            clusters[nearest_medoid_index].append(point)
        else:
            clusters[nearest_medoid_index] = [point]
    return clusters


def calculate_total_cost(clusters, medoids):
    total_cost = 0
    for medoid_idx, points in clusters.items():
        for point in points:
            total_cost += euclidean_distance(point, medoids[medoid_idx])
    return total_cost


def k_medoids(data, k):
    # Losowe wybranie k punktów jako początkowe medoidy
    initial_medoids_indices = np.random.choice(len(data), k, replace=False)
    medoids = [data[index] for index in initial_medoids_indices]

    print(f"Początkowe medoidy: {medoids}")

    clusters = assign_points_to_medoids(data, medoids)
    current_cost = calculate_total_cost(clusters, medoids)

    while True:
        best_cost = current_cost
        best_medoids = medoids.copy()
        best_clusters = clusters

        for i, medoid in enumerate(medoids):
            for point in data:
                if any(np.array_equal(point, m) for m in medoids):
                    continue

                temp_medoids = medoids.copy()
                temp_medoids[i] = point
                temp_clusters = assign_points_to_medoids(data, temp_medoids)
                temp_cost = calculate_total_cost(temp_clusters, temp_medoids)

                if temp_cost < best_cost:
                    best_cost = temp_cost
                    best_medoids = temp_medoids.copy()
                    best_clusters = temp_clusters

        if best_cost < current_cost:
            medoids = best_medoids
            clusters = best_clusters
            current_cost = best_cost
            print(f"Nowe medoidy: {medoids} z kosztem: {current_cost}")
        else:
            print("\nMedoidy przestały się zmieniać, kończymy iteracje.")
            break

    labels = np.zeros(len(data))
    for cluster_index, points in clusters.items():
        for point in points:
            point_index = np.where(np.all(data == point, axis=1))[0][0]
            labels[point_index] = cluster_index

    return medoids, labels, clusters


def generate_random_points(min_x, max_x, min_y, max_y, n):
    x_coords = np.random.randint(min_x, max_x + 1, size=n)
    y_coords = np.random.randint(min_y, max_y + 1, size=n)
    return np.column_stack((x_coords, y_coords))

def plot_clusters(data, medoids, clusters):
    plt.figure(figsize=(5, 5))  # Zwiększamy rozmiar wyjściowy wykresu

    # Definicja kolorów dla klastrów (możesz dostosować lub rozszerzyć w razie potrzeby)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.8']

    for cluster_index, points in clusters.items():
        medoid = medoids[cluster_index]
        cluster_color = colors[cluster_index % len(colors)]

        # Punkty w klastrze
        cluster_points = np.array(points)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_color, label=f'Cluster {cluster_index}',
                    marker='o')

        # Medoid dla klastra jako kwadrat
        plt.scatter(medoid[0], medoid[1], marker='s', c=cluster_color, edgecolors='k', s=100,
                    label=f'Medoid {cluster_index}')

    plt.title('K-medoids Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Ustawienie legendy poza wykres
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')

    plt.grid(True)

    # Ustawienie identycznej skali dla osi X i Y
    plt.axis('equal')

    # Zapisz wykres do pliku PNG z większym rozmiarem i rozdzielczością
    plt.savefig('k_medoids_clusters.png', bbox_inches='tight', dpi=300)

    plt.show()


def run_k_medoids_test(data, k):
    medoids, labels, clusters = k_medoids(data, k)

    print("\nOstateczne medoidy i przypisania punktów:")
    for cluster_index, medoid in enumerate(medoids):
        print(f"Medoid dla etykiety {cluster_index}: {medoid}")
        print(f"Punkty przypisane do etykiety {cluster_index}: {clusters[cluster_index]}")

    print("\nEtykiety dla każdego punktu:")
    for point, label in zip(data, labels):
        print(f"Punkt {point} przypisany do etykiety {int(label)}")

    # Wykres
    plot_clusters(data, medoids, clusters)
data = np.array([
    [1, 2], [2, 3], [3, 4], [8, 7], [7, 8], [9, 7]
])
k = 2

run_k_medoids_test(data, k)
# Przykład użycia dla 10 punktów
data_10 = generate_random_points(0, 10, 0, 10, 10)
k_10 = 3

run_k_medoids_test(data_10, k_10)
# Przykład użycia dla 50 punktów
data_50 = generate_random_points(0, 50, 0, 50, 50)
k_50 = 5

run_k_medoids_test(data_50, k_50)
# Przykład użycia dla 100 punktów
data_100 = generate_random_points(0, 100, 0, 100, 100)
k_100 = 10


run_k_medoids_test(data_100, k_100)

