import TAT
import numpy as np

Tensor = TAT(float)


def get_dimension(tensor, name):
    return tensor.edge[tensor.name.index(name)]


def get_dense_matrix(total_length, point_list, get_site):
    point_map = {}
    for i, p in enumerate(point_list):
        point_map[p] = i
    Dh = get_dimension(get_site(0), "L")
    Dt = get_dimension(get_site(total_length - 1), "R")
    result = Tensor(["R", "R'"], [Dh, Dh]).identity({("R", "R'")})
    tail = Tensor(["L", "L'"], [Dt, Dt]).identity({("L", "L'")})
    for i in range(total_length):
        site = get_site(i)
        if i in point_map:
            index = point_map[i]
            P1 = "P" + str(index)
            P2 = P1 + "'"
            result = result.contract(site.edge_rename({"P": P1}),
                                     {("R", "L")}).contract(
                                         site.edge_rename({
                                             "P": P2,
                                             "R": "R'"
                                         }), {("R'", "L")})
        else:
            result = result.contract(site, {("R", "L")}).contract(
                site.edge_rename({"R": "R'"}), {("R'", "L"), ("P", "P")})
        result /= result.norm_max()
    result = result.contract(tail, {("R", "L"), ("R'", "L'")})
    result /= result.norm_max()
    return result


def get_merged_rho(rho, part={0}):
    copart = {i for i in range(len(rho.name) // 2) if i not in part}
    res = rho.merge_edge({
        "I1": ["P" + str(i) for i in part],
        "O1": ["P" + str(i) + "'" for i in part],
        "I2": ["P" + str(i) for i in copart],
        "O2": ["P" + str(i) + "'" for i in copart]
    })
    return res
