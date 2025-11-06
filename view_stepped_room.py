import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Load points from the text file ===
filename = "room_irregular_surface.txt"
points = []

with open(filename) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith(("#", "END")):
            x, y, z = map(float, line.split(","))
            points.append((x, y, z))

# === Separate into x, y, z lists ===
xs = [p[0] for p in points]
ys = [p[1] for p in points]
zs = [p[2] for p in points]

# === Plot setup ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, c='red', s=50, label="Vertices")

# Connect points roughly by order (simple wireframe)
for i in range(len(points) - 1):
    x1, y1, z1 = points[i]
    x2, y2, z2 = points[i + 1]
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'b--')

# === Axes labels and title ===
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Height Z (m)')
ax.set_title('Stepped Floor: 6m × 4m Room')
ax.legend()

ax.view_init(elev=25, azim=45)
plt.tight_layout()
plt.show()
