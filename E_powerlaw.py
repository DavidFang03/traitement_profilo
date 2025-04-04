import numpy as np
import matplotlib.pyplot as plt
import json
import A_utilit as utilit


json_path = "DATA.json"
datas = []
with open(json_path, "r") as infile:
    try:
        datas = json.loads(infile.read())
    except json.JSONDecodeError:
        raise Exception(f"{json_path} is empty or not a valid JSON file")


our_mass = np.array([0, 0, 89.46, 0, 63.7, 0, 49.59,
                    0, 32.63, 0, 23.81, 0, 16.69])

Zc = []
Bottom = []
M = []
H = []
E = []
g = 9.81

for data in datas:
    zc = np.abs(data["zc"])
    bottom = np.abs(data["bottom"])
    m_nb = int(data["mass"])
    m = our_mass[m_nb]
    h = data["height"]
    e = (m*1e-3)*g*h

    Zc.append(zc)
    Bottom.append(bottom)
    M.append(m)
    H.append(h)
    E.append(e)

    plt.text(e, zc, f"{m_nb:.1f} : {m:.1f}g", fontsize=9, ha='right')

E = np.array(M) * 1e-3 * g * np.array(H)

p, cov = np.polyfit(np.log(E), np.log(Zc), 1, cov=True)
a, b = p
print(f"p = {p}, cov = {cov}")

p2, cov2 = np.polyfit(np.log(E), np.log(Bottom), 1, cov=True)
a2, b2 = p2
print(f"p = {p2}, cov = {cov2}")

fig, ax = plt.subplots(2)

ax[0].plot(np.log(E), np.log(Zc), "x")
ax[0].plot(np.log(E), a*np.log(E)+b, label=f"{a:.2f}")
ax[1].plot(np.log(E), np.log(Bottom), "x")
ax[1].plot(np.log(E), a2*np.log(E)+b2, label=f"{a2:.2f}")

ax[0].set_xlabel("log(Energy) (J)")
ax[0].set_ylabel("log(Zc) (px)")
ax[0].set_title("zc(E) (loglog)")

ax[1].set_xlabel("log(Energy) (J)")
ax[1].set_ylabel("log(Bottom) (px)")
ax[1].set_title("Bottom(E) (loglog)")

ax[0].legend()
ax[1].legend()

# plt.legend()
plt.show()
