import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

#result 
g_head = 0.30
g_distance = 0.23
g_body = 0.257
g_leg = 0.152

#mathplot bokeh linearcolormap like kde
cmap = get_cmap("Reds")         
norm = Normalize(vmin=0, vmax=0.293)

def get_color(val):
    return cmap(norm(val))

fig, ax = plt.subplots(figsize=(4, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")

head  = patches.Circle((5, 18), 1.2, edgecolor="black", facecolor=get_color(g_head))
torso = patches.Rectangle((3.5, 10), 3, 6, edgecolor="black", facecolor=get_color(g_body))
larm  = patches.Rectangle((2.5, 10), 1.2, 4, edgecolor="black", facecolor=get_color(g_distance))
rarm  = patches.Rectangle((6.3, 10), 1.2, 4, edgecolor="black", facecolor=get_color(g_distance))
lleg  = patches.Rectangle((4, 4), 1, 6, edgecolor="black", facecolor=get_color(g_leg))
rleg  = patches.Rectangle((5, 4), 1, 6, edgecolor="black", facecolor=get_color(g_leg))

for p in [head, torso, larm, rarm, lleg, rleg]:
    ax.add_patch(p)

plt.text(5, 19.3, "KO Strike Heatmap", ha="center", fontsize=15, weight="bold")
plt.text(5, 17.0, f"Head Impact: {g_head:.3f}", ha="center", fontsize=10)
plt.text(5, 14.5, f"Distance/Arm Impact: {g_distance:.3f}", ha="center", fontsize=10)
plt.text(5, 9.5,  f"Body Impact: {g_body:.3f}", ha="center", fontsize=10)
plt.text(5, 3.0,  f"Leg Impact: {g_leg:.3f}", ha="center", fontsize=10)

#colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Effect Size (Hedges g)", fontsize=10)

plt.tight_layout()
plt.savefig("ko_heatmap_hedges.png", dpi=300, bbox_inches='tight')
plt.show()