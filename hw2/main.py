import pygame
import random

# Функция для вычисления Евклидова расстояния между двумя точками
def dist(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

# Функция для поиска соседей точки в пределах радиуса eps
def region_query(data, point, eps):
    neighbors = []
    for neighbor in data:
        if dist(point, neighbor) < eps:
            neighbors.append(neighbor)
    return neighbors

# Функция для расширения кластера
def expand_cluster(data, point, neighbors, cluster_id, eps, min_pts, labels):
    labels[data.index(point)] = cluster_id
    for neighbor in neighbors:
        if labels[data.index(neighbor)] == -1:
            labels[data.index(neighbor)] = cluster_id
            new_neighbors = region_query(data, neighbor, eps)
            if len(new_neighbors) >= min_pts:
                neighbors += new_neighbors
        elif labels[data.index(neighbor)] == 0:
            labels[data.index(neighbor)] = cluster_id

# Основная функция DBSCAN
def dbscan(data, eps, min_pts):
    cluster_id = 0
    labels = [0] * len(data)
    for point in data:
        if labels[data.index(point)] == 0:
            neighbors = region_query(data, point, eps)
            if len(neighbors) < min_pts:
                labels[data.index(point)] = -1
            else:
                cluster_id += 1
                expand_cluster(data, point, neighbors, cluster_id, eps, min_pts, labels)
    return labels

# Функция для назначения цвета точкам в зависимости от кластера
def add_color(color):
    if color == -1:
        return RED
    else:
        random.seed(color)  # Обеспечить одинаковый цвет для одного cluster_id
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

# Основные параметры окна и цветов
WIDTH = 360
HEIGHT = 480
FPS = 30
WHITE = [255, 255, 255]
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Инициализация Pygame
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("HOMEWORK")
clock = pygame.time.Clock()
points = []
eps = 30
min_pts = 2

running = True

while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                point = event.pos
                points.append(point)
                pygame.draw.circle(screen, WHITE, point, 5)
                pygame.display.flip()  # Обновить экран после рисования

        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                labels = dbscan(points, eps, min_pts)
                print(labels, points)
                screen.fill(BLACK)  # Очистить экран перед перерисовкой точек
                for point, label in zip(points, labels):
                    color = add_color(label)
                    pygame.draw.circle(screen, color, point, 5)
                pygame.display.flip()  # Обновить экран после рисования

pygame.quit()
