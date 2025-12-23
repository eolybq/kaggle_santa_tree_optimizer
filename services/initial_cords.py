import numpy as np
from shapely.strtree import STRtree
from decimal import Decimal, getcontext
import pandas as pd

from services.chtree import ChristmasTree, scale_factor


getcontext().prec = 25


def save_cords(df, filename):
    initial_df = df.copy()
    # To ensure everything is kept as a string, prepend an 's'
    for col in df.columns:
        if col != 'id':
            initial_df[col] = 's' + initial_df[col].astype('string')
    initial_df.to_csv(f'{filename}.csv', index=False)
    print(f"Data saved to {filename}.csv")


def place_tree(placed_trees, min_bounds, min_side, attempts):
    if not placed_trees:
        return (0, 0, 0), ChristmasTree(Decimal(0), Decimal(0), Decimal(0))
    placed_polygons = [t.polygon for t in placed_trees]
    tree_index = STRtree(placed_polygons)

    for attempt in range(attempts):
        if attempt % 100 == 0:
            padding = attempt // 100
        else:
            padding = 0

        x_cand = np.random.normal(0, min_side//4 + padding//2)
        y_cand = np.random.normal(0, min_side//4 + padding//2)
        deg_cand = np.random.randint(0,360)

        cords = (Decimal(x_cand), Decimal(y_cand), Decimal(deg_cand))

        cand_tree = ChristmasTree(*cords)


        cand_poly = cand_tree.polygon

        possible_indices = tree_index.query(cand_poly)

        if not any((cand_poly.intersects(placed_polygons[i]) and not cand_poly.touches(placed_polygons[i])) for i in possible_indices):
            bounds = [Decimal(t) / scale_factor for t in cand_tree.polygon.bounds]
            for idx in range(4):
                if bounds[idx] < min_bounds[idx]:
                    min_bounds[idx] = bounds[idx]
            min_side = max(min_bounds[2] - min_bounds[0], min_bounds[3] - min_bounds[1])

            return cords, cand_tree

    print("Out of attempts")
    return (0, 0, 0), ChristmasTree(Decimal(0), Decimal(0), Decimal(0))


def main(N):
    placed_trees = []
    cords_trees = []
    for i in range(1, N + 1):
        min_bounds = np.array([-0.425, -0.2, 0.425, 0.8])
        min_side = 1
        for j in range(i):
            print("i, j", i, j)
            cords, tree = place_tree(placed_trees, min_bounds, min_side, attempts=100000)
            if cords and tree:
                placed_trees.append(tree)
                cords_trees.append(cords)
        placed_trees = []
        

    index = [f'{n:03d}_{t}' for n in range(1, N + 1) for t in range(n)]
    df = pd.DataFrame(cords_trees, index=index, columns=['x', 'y', 'deg'])
    df.reset_index(inplace=True, names='id')

    save_cords(df)
    return df


if __name__ == "__main__":
    df = main(N=200)