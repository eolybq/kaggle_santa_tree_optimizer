import pandas as pd
import numpy as np
import math
import random
import copy
import os
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity

# --- TVOJE DEFINICE ---
getcontext().prec = 25
scale_factor = Decimal('1e15')

class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w, trunk_h = Decimal('0.15'), Decimal('0.2')
        base_w, mid_w, top_w = Decimal('0.7'), Decimal('0.4'), Decimal('0.25')
        tip_y, tier_1_y, tier_2_y, base_y = Decimal('0.8'), Decimal('0.5'), Decimal('0.25'), Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon([
            (Decimal('0.0') * scale_factor, tip_y * scale_factor),
            (top_w / 2 * scale_factor, tier_1_y * scale_factor),
            (top_w / 4 * scale_factor, tier_1_y * scale_factor),
            (mid_w / 2 * scale_factor, tier_2_y * scale_factor),
            (mid_w / 4 * scale_factor, tier_2_y * scale_factor),
            (base_w / 2 * scale_factor, base_y * scale_factor),
            (trunk_w / 2 * scale_factor, base_y * scale_factor),
            (trunk_w / 2 * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / 2) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / 2) * scale_factor, base_y * scale_factor),
            (-(base_w / 2) * scale_factor, base_y * scale_factor),
            (-(mid_w / 4) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / 2) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / 4) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / 2) * scale_factor, tier_1_y * scale_factor),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))
        self.bbox = self.polygon.bounds

def save_cords(df, filename):
    initial_df = df.copy()
    for col in df.columns:
        if col != 'id':
            initial_df[col] = 's' + initial_df[col].astype('string')
    initial_df.to_csv(f'{filename}.csv', index=False)
    print(f"Data saved to {filename}.csv")

# --- OPTIMALIZAČNÍ FUNKCE ---

def calculate_score(trees):
    if not trees: return 0.0
    min_x = min(t.bbox[0] for t in trees)
    min_y = min(t.bbox[1] for t in trees)
    max_x = max(t.bbox[2] for t in trees)
    max_y = max(t.bbox[3] for t in trees)
    # Skóre počítáme v měřítku 1:1 (vydělíme scale_factor)
    side = max(max_x - min_x, max_y - min_y) / float(scale_factor)
    return side * side

def calculate_score_incremental(trees, old_bounds, old_tree_bbox, new_tree_bbox):
    min_x, min_y, max_x, max_y = old_bounds
    # Pokud byl starý strom na hranici, musíme přepočítat vše
    if (old_tree_bbox[0] <= min_x + 1e-7 or old_tree_bbox[1] <= min_y + 1e-7 or
            old_tree_bbox[2] >= max_x - 1e-7 or old_tree_bbox[3] >= max_y - 1e-7):
        min_x = min(t.bbox[0] for t in trees)
        min_y = min(t.bbox[1] for t in trees)
        max_x = max(t.bbox[2] for t in trees)
        max_y = max(t.bbox[3] for t in trees)
    else:
        # Jinak jen rozšíříme hranice o nový strom
        min_x = min(min_x, new_tree_bbox[0])
        min_y = min(min_y, new_tree_bbox[1])
        max_x = max(max_x, new_tree_bbox[2])
        max_y = max(max_y, new_tree_bbox[3])

    side = max(max_x - min_x, max_y - min_y) / float(scale_factor)
    return side * side, (min_x, min_y, max_x, max_y)

def main(n_limit=200, iterations_per_temp=50):
    all_optimized_data = []

    # Postupujeme od N=1 do 200
    for n in range(1, n_limit + 1):
        print(f"\n--- Optimizing N={n} ---")

        # Inicializace: pro jednoduchost mřížka (lze nahradit tvým init_trees)
        current_trees = []
        rows = int(math.sqrt(n)) + 1
        for i in range(n):
            current_trees.append(ChristmasTree(str(i % rows), str(i // rows), '0'))


        # SA Parametry
        T, Tmin, alpha = 1.0, 0.001, 0.99
        cur_score = calculate_score(current_trees)
        min_x = min(t.bbox[0] for t in current_trees)
        min_y = min(t.bbox[1] for t in current_trees)
        max_x = max(t.bbox[2] for t in current_trees)
        max_y = max(t.bbox[3] for t in current_trees)
        cur_bounds = (min_x, min_y, max_x, max_y)

        while T > Tmin:
            for _ in range(iterations_per_temp):
                idx = random.randint(0, n - 1)
                old_tree = current_trees[idx]
                old_state = (old_tree.center_x, old_tree.center_y, old_tree.angle, old_tree.polygon, old_tree.bbox)

                # Náhodný posun
                new_x = old_tree.center_x + Decimal(str(random.uniform(-0.1, 0.1)))
                new_y = old_tree.center_y + Decimal(str(random.uniform(-0.1, 0.1)))
                new_ang = (old_tree.angle + Decimal(str(random.uniform(-10, 10)))) % 360

                # Update
                current_trees[idx] = ChristmasTree(str(new_x), str(new_y), str(new_ang))

                # Kolize
                if any(current_trees[idx].polygon.intersects(current_trees[i].polygon) for i in range(n) if i != idx):
                    current_trees[idx] = old_tree # Revert
                    continue

                # Skóre
                new_score, new_bounds = calculate_score_incremental(current_trees, cur_bounds, old_state[4], current_trees[idx].bbox)
                diff = new_score - cur_score

                if diff < 0 or random.random() < math.exp(-diff / T):
                    cur_score, cur_bounds = new_score, new_bounds
                else:
                    current_trees[idx] = old_tree # Revert

            T *= alpha

        print(f"Final side for N={n}: {math.sqrt(cur_score):.6f}")

        # Sběr dat pro export
        for i, t in enumerate(current_trees):
            all_optimized_data.append({
                'id': f'{n:03d}_{i}',
                'x': str(t.center_x),
                'y': str(t.center_y),
                'deg': str(t.angle)
            })

    # Uložení výsledků
    final_df = pd.DataFrame(all_optimized_data)
    save_cords(final_df, "test_test_test")

if __name__ == "__main__":
    main(n_limit=200) # Zkus nejdřív pro malý počet (např. 10)