import copy
import pandas as pd
import numpy as np
from shapely.strtree import STRtree
from decimal import Decimal, getcontext
from math import exp

from services.chtree import ChristmasTree, scale_factor
from services.initial_cords import save_cords

 
getcontext().prec = 25


def convert_to_np(df):
    for col in ['x', 'y', 'deg']:
        df[col] = df[col].apply(lambda x: x.replace("s", "")).apply(Decimal)

    df['group_id'] = df['id'].str.split('_').str[0].astype(int)
    groups = df.groupby('group_id')

    data = []
    for group_id, group in groups:
        row = group[['x', 'y', 'deg']].to_numpy()
        data.append(row)
    return data


def compute_side(placed_trees):
    min_x = min(p.bounds[0] for p in placed_trees) / float(scale_factor)
    max_x = max(p.bounds[2] for p in placed_trees) / float(scale_factor)
    min_y = min(p.bounds[1] for p in placed_trees) / float(scale_factor)
    max_y = max(p.bounds[3] for p in placed_trees) / float(scale_factor)

    return max(max_x - min_x, max_y - min_y)



def compute_loss(cand_tree, new_tree, tree_index, placed_trees, idx, eta_distance=5, eta_origin=0.1, eta_n=0.01):
    possible_indices = tree_index.query(cand_tree.polygon, predicate='dwithin', distance=eta_distance)

    old_distances = 0
    for close_tree in possible_indices:
        old_distances += cand_tree.polygon.distance(placed_trees[close_tree].polygon)
    old_distance_from_origin = cand_tree.polygon.distance(ChristmasTree(Decimal(0), Decimal(0), Decimal(0)).polygon)


    # Pohyb stromu -> nový list
    new_placed_trees = copy.deepcopy(placed_trees)
    new_placed_trees[idx] = new_tree

    new_tree_index = STRtree([x.polygon for x in new_placed_trees])
    new_possible_indices = new_tree_index.query(new_tree.polygon, predicate='dwithin', distance=eta_distance)
    
    new_distances = 0
    for close_tree in new_possible_indices:
        new_distances += new_tree.polygon.distance(new_placed_trees[close_tree].polygon)

    new_distance_from_origin = new_tree.polygon.distance(ChristmasTree(Decimal(0), Decimal(0), Decimal(0)).polygon)

    # NOTE unused loss
    old_loss = Decimal(old_distances/len(placed_trees)) + Decimal(eta_origin*old_distance_from_origin) - Decimal(eta_n*len(possible_indices))
    new_loss = Decimal(new_distances/len(new_placed_trees)) + Decimal(eta_origin*new_distance_from_origin) - Decimal(eta_n*len(new_possible_indices))


    # old_bbox = compute_side([p.polygon for p in placed_trees])**2 / len(placed_trees)
    # new_bbox = compute_side([p.polygon for p in new_placed_trees])**2 / len(new_placed_trees)
    #
    #
    # old_loss = Decimal(old_bbox) + Decimal(eta_origin*old_distance_from_origin) / scale_factor
    # new_loss = Decimal(new_bbox) + Decimal(eta_origin*new_distance_from_origin) / scale_factor

    # print(f"Old: {old_loss}, New: {new_loss}, Improvement: {old_loss - new_loss}")

    return {"old_loss": old_loss, "new_loss": new_loss, "new_trees_list": new_placed_trees}
    
    
def candidate(temp, cand_tree, new_tree, tree_index, placed_trees, idx, eta_distance=5*scale_factor):
    new_placed_trees = copy.deepcopy(placed_trees)
    new_placed_trees[idx] = new_tree

    new_tree_index = STRtree([x.polygon for x in new_placed_trees])
    new_possible_indices = new_tree_index.query(new_tree.polygon, predicate='dwithin', distance=eta_distance)
    # print(f"idx:{idx}, Possible indices for collision check: {new_possible_indices}")


    if any((new_tree.polygon.intersects(new_placed_trees[i].polygon) and not new_tree.polygon.touches(new_placed_trees[i].polygon)) for i in new_possible_indices if i != idx):
        # print("Veto - kolize stromů")
        return placed_trees

    loss_dict = compute_loss(cand_tree, new_tree, tree_index, placed_trees, idx)
    # print(f"Old loss: {loss_dict["old_loss"]}, New loss: {loss_dict["new_loss"]}")
    #print((loss_dict["new_loss"] - loss_dict["old_loss"]))

    if loss_dict["old_loss"] > loss_dict["new_loss"]:
        return loss_dict["new_trees_list"]

    # P = exp(-(loss_dict["new_loss"] - loss_dict["old_loss"]) / Decimal(temp))
    # print(f"Acceptance probability: {P}")
    #
    #
    # if np.random.uniform() < P:
    #     return loss_dict["new_trees_list"]
    else:
        return placed_trees


def main(save_file, N=None, file_to_optimize="initial_coordinates", step_deg=5, step_sd = 0.2, default_temperature=1000, cooling_rate=0.996, min_temperature=0.01):
    initial_df = pd.read_csv(f'data/{file_to_optimize}.csv')
    data = convert_to_np(initial_df)

    if N is None:
        N = len(data)

    for n in range(N):

        placed_trees = [ChristmasTree(*t) for t in data[n]]
        temperature = default_temperature
        # while temperature > min_temperature:
        for iteration in range(10000):
            tree_index = STRtree([x.polygon for x in placed_trees])

            rnd_selection = int(np.random.uniform(0, 1) * (n + 1))
            cand_tree = placed_trees[rnd_selection]

            new_tree = ChristmasTree(
                cand_tree.center_x + Decimal(np.random.normal(0, step_sd)),
                cand_tree.center_y + Decimal(np.random.normal(0, step_sd)),
                cand_tree.angle + np.random.randint(-step_deg, step_deg)
            )
            # TODO Vracet jen idx new_tree a měnit jen data[n][idx] -> optimalizovat
            placed_trees = candidate(temp=temperature, cand_tree=cand_tree, new_tree=new_tree, tree_index=tree_index, placed_trees=placed_trees, idx=rnd_selection)
            # print(f"Swapped placed trees: {swap_placed_trees}")
            # temperature *= cooling_rate

        data[n] = [(t.center_x, t.center_y, t.angle) for t in placed_trees]
        print(n)
            # tree_objects_list[n] = swap_placed_trees
            




    flatten_data = [item for sublist in data for item in sublist]

    # print(len(flatten_data))
    index = [f'{n:03d}_{t}' for n in range(1, len(data) + 1) for t in range(n)]
    df = pd.DataFrame(flatten_data, index=index, columns=['x', 'y', 'deg'])
    df.reset_index(inplace=True, names='id')

    save_cords(df, save_file)


if __name__ == "__main__":
    main(N=200, file_to_optimize="optimized", save_file="optimized_final")
