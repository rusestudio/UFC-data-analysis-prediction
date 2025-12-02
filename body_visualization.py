import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

g_body = 0.258
g_leg  = 0.063

# to %
body_pct = (g_body / 0.3) * 100
leg_pct  = (g_leg / 0.3) * 100

body_pct = max(0, min(body_pct, 100))
leg_pct  = max(0, min(leg_pct, 100))

def norm(x):
    return min(max(x / 0.3, 0), 1)

body_intensity = norm(g_body)
leg_intensity  = norm(g_leg)


def heat_color(intensity, base=(1,0,0)):
    """
    Blend between white → red based on intensity.
    """
    r = base[0] * intensity + 1*(1-intensity)
    g = base[1] * intensity + 1*(1-intensity)
    b = base[2] * intensity + 1*(1-intensity)
    return (r,g,b)

body_color = heat_color(body_intensity, base=(1,0,0))   # torso
leg_color  = heat_color(leg_intensity, base=(1,0.2,0.2)) # legs 

fig, ax = plt.subplots(figsize=(4, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")

# head
head = patches.Circle((5, 18), 1.2, edgecolor='black', facecolor="#ffffff")
ax.add_patch(head)

# torso
torso = patches.Rectangle((3.5, 10), 3, 6, edgecolor='black', facecolor=body_color)
ax.add_patch(torso)

# leg
left_leg = patches.Rectangle((4, 4), 1, 6, edgecolor='black', facecolor=leg_color)
right_leg = patches.Rectangle((5, 4), 1, 6, edgecolor='black', facecolor=leg_color)
ax.add_patch(left_leg)
ax.add_patch(right_leg)

# arm
left_arm = patches.Rectangle((2.5, 10), 1, 4, edgecolor='black', facecolor="#f2f2f2")
right_arm = patches.Rectangle((6.5, 10), 1, 4, edgecolor='black', facecolor="#f2f2f2")
ax.add_patch(left_arm)
ax.add_patch(right_arm)

#label
plt.text(5, 19.3, "KO Strike Heatmap", ha='center', fontsize=15, weight='bold')
plt.text(5, 9.4, f"Body KO Impact: {body_pct:.1f}%", ha='center', fontsize=10)
plt.text(5, 3.2, f"Leg KO Impact: {leg_pct:.1f}%", ha='center', fontsize=10)
plt.show()
