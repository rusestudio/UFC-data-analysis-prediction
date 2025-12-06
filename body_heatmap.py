import matplotlib.pyplot as plt
import matplotlib.patches as patches

#result
g_head = 0.30          # strongest
g_distance = 0.23      # arms / upper body
g_body = 0.257         # torso
g_leg = 0.152          # legs

# normalize to 0.3
def norm(x):
    return min(max(x / 0.30, 0), 1)

n_head = norm(g_head)
n_dist = norm(g_distance)
n_body = norm(g_body)
n_leg  = norm(g_leg)

p_head = n_head * 100
p_dist = n_dist * 100
p_body = n_body * 100
p_leg  = n_leg * 100

#color grad
def heat_color(intensity):
    boosted = intensity ** 3.0  
    return (1,
            1 - boosted,
            1 - boosted)

head_color = heat_color(n_head)
arm_color  = heat_color(n_dist)
body_color = heat_color(n_body)
leg_color  = heat_color(n_leg)

fig, ax = plt.subplots(figsize=(4, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")

head = patches.Circle((5, 18), 1.2, edgecolor='black', facecolor=head_color)
ax.add_patch(head)

torso = patches.Rectangle((3.5, 10), 3, 6, edgecolor='black', facecolor=body_color)
ax.add_patch(torso)

left_arm = patches.Rectangle((2.5, 10), 1.2, 4, edgecolor='black', facecolor=arm_color)
right_arm = patches.Rectangle((6.3, 10), 1.2, 4, edgecolor='black', facecolor=arm_color)
ax.add_patch(left_arm)
ax.add_patch(right_arm)

left_leg = patches.Rectangle((4, 4), 1, 6, edgecolor='black', facecolor=leg_color)
right_leg = patches.Rectangle((5, 4), 1, 6, edgecolor='black', facecolor=leg_color)
ax.add_patch(left_leg)
ax.add_patch(right_leg)

plt.text(5, 19.3, "KO Strike Heatmap", ha='center', fontsize=15, weight='bold')
plt.text(5, 17,   f"Head Impact: {p_head:.1f}%", ha='center', fontsize=10)
plt.text(5, 14.5, f"Distance/Arm Impact: {p_dist:.1f}%", ha='center', fontsize=10)
plt.text(5, 9.5,  f"Body Impact: {p_body:.1f}%", ha='center', fontsize=10)
plt.text(5, 3,    f"Leg Impact: {p_leg:.1f}%", ha='center', fontsize=10)

plt.savefig("ko_heatmap_perct.png", dpi=300, bbox_inches='tight')
plt.show()
