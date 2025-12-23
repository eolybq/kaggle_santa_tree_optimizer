import pandas as pd
import numpy as np
import plotly.graph_objects as go
from decimal import getcontext, Decimal


from chtree import ChristmasTree, scale_factor

getcontext().prec = 25


def plot_n_trees_plotly(n_count, filename):
    """
    Interaktivní vykreslení stromů pro konkrétní N pomocí Plotly.
    """

    df = pd.read_csv(f"{filename}.csv")
    for col in ['x', 'y', 'deg']:
        df[col] = df[col].apply(lambda x: x.replace("s", "")).apply(Decimal)
    # 1. Filtrace dat
    n_prefix = f"{n_count:03d}_"
    subset = df[df['id'].str.startswith(n_prefix)].copy()

    if subset.empty:
        print(f"Žádná data pro N={n_count} nebyla nalezena.")
        return

    fig = go.Figure()

    all_polys = []

    # 2. Přidání stromů do grafu
    for _, row in subset.iterrows():
        # Vytvoření instance (použije tvou třídu)
        tree = ChristmasTree(row['x'], row['y'], row['deg'])
        poly = tree.polygon
        all_polys.append(poly)

        # Normalizace souřadnic pro zobrazení
        x_coords, y_coords = poly.exterior.xy
        x_plot = np.array(x_coords) / float(scale_factor)
        y_plot = np.array(y_coords) / float(scale_factor)

        # Přidání stromečku jako stínované plochy
        fig.add_trace(go.Scatter(
            x=x_plot.tolist(),
            y=y_plot.tolist(),
            fill="toself",
            mode='lines',
            name=row['id'],
            text=f"ID: {row['id']}<br>Deg: {row['deg']}",
            line=dict(color='black', width=1),
            fillcolor='rgba(45, 90, 39, 0.5)', # Průhledná zelená
            hoveron='fills'
        ))

    # 3. Výpočet a vykreslení Bounding Boxu
    min_x = min(p.bounds[0] for p in all_polys) / float(scale_factor)
    max_x = max(p.bounds[2] for p in all_polys) / float(scale_factor)
    min_y = min(p.bounds[1] for p in all_polys) / float(scale_factor)
    max_y = max(p.bounds[3] for p in all_polys) / float(scale_factor)

    side = max(max_x - min_x, max_y - min_y)

    # Čtvercová ohrada
    fig.add_shape(
        type="rect",
        x0=min_x, y0=min_y, x1=min_x + side, y1=min_y + side,
        line=dict(color="Red", width=2, dash="dash"),
    )

    # 4. Nastavení vzhledu grafu
    fig.update_layout(
        title=f"Interaktivní vizualizace N={n_count} (Plocha: {side**2:.6f})",
        xaxis=dict(title="X", scaleanchor="y", scaleratio=1), # Zajišťuje poměr stran 1:1
        yaxis=dict(title="Y"),
        showlegend=True,
        width=800,
        height=800,
        template="plotly_white"
    )

    fig.show()








if __name__ == "__main__":
    while True:
        inpt = int(input("zadej cylso"))
        plot_n_trees_plotly(inpt, filename="optimized")