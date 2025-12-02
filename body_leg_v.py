import matplotlib.pyplot as plt
import matplotlib.patches as patches

g_body = 0.258
g_leg = 0.063

body_pct = min(max((g_body / 0.3) * 100, 0), 100)
leg_pct  = min(max((g_leg  / 0.3) * 100, 0), 100)

def norm(x):
    return min(max(x / 0.3, 0), 1)

body_intensity = norm(g_body)
leg_intensity  = norm(g_leg)

def heat_color(intensity, base=(1,0,0)):
    r = base[0] * intensity + 1*(1-intensity)
    g = base[1] * intensity + 1*(1-intensity)
    b = base[2] * intensity + 1*(1-intensity)
    return (r,g,b)

body_color = heat_color(body_intensity)
leg_color  = heat_color(leg_intensity, base=(1,0.2,0.2))

def draw_figure(mode="body"):
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")

    # head
    head = patches.Circle((5, 18), 1.2, edgecolor='black', facecolor="#ffffff")
    ax.add_patch(head)

    # torrs
    torso_color = body_color if mode == "body" else "#ffffff"
    torso = patches.Rectangle((3.5, 10), 3, 6, edgecolor='black', facecolor=torso_color)
    ax.add_patch(torso)

    #leg
    leg_color_set = leg_color if mode == "leg" else "#ffffff"
    left_leg = patches.Rectangle((4, 4), 1, 6, edgecolor='black', facecolor=leg_color_set)
    right_leg = patches.Rectangle((5, 4), 1, 6, edgecolor='black', facecolor=leg_color_set)
    ax.add_patch(left_leg)
    ax.add_patch(right_leg)

    # arm
    left_arm = patches.Rectangle((2.5, 10), 1, 4, edgecolor='black', facecolor="#ffffff")
    right_arm = patches.Rectangle((6.5, 10), 1, 4, edgecolor='black', facecolor="#ffffff")
    ax.add_patch(left_arm)
    ax.add_patch(right_arm)

    #label
    plt.text(5, 19.3, "KO Strike Heatmap", ha='center', fontsize=15, weight='bold')

    if mode == "body":
        plt.text(5, 9.2, f"Body KO Impact: {body_pct:.1f}%", ha='center', fontsize=11)
    elif mode == "leg":
        plt.text(5, 3, f"Leg KO Impact: {leg_pct:.1f}%", ha='center', fontsize=11)

    return fig


#bdy
fig1 = draw_figure(mode="body")
fig1.savefig("body.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

#leg
fig2 = draw_figure(mode="leg")
fig2.savefig("leg.png", dpi=300, bbox_inches='tight')
plt.close(fig2)
