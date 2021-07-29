#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import TAT
import numpy as np

Tensor = TAT(float)


def prepare_L_or_R(*, A, l, r):
    D = get_dimension(A, l)
    res = Tensor([r, l], [D, D]).randn()
    for t in range(1000):
        res = res.contract(A, {(r, l)})
        res /= res.norm_max()
        res = res.merge_edge({l: [l, "P"]})
        _, res = res.qr('r', {r}, r, l)
    return res


def inv(tensor):
    block = tensor.block[{}]
    new_block = np.linalg.inv(block)
    res = tensor.same_shape()
    res.block[{}] = new_block
    return res


def prepare_input(*, D):
    A = Tensor(["L", "R", "P"], [D, D, 2]).randn()
    L = prepare_L_or_R(A=A, l="L", r="R")
    R = prepare_L_or_R(A=A, l="R", r="L")
    invL = inv(L)
    invR = inv(R)
    AL = L.contract(A, {("R", "L")}).contract(invL, {("R", "L")})
    AR = R.contract(A, {("L", "R")}).contract(invR, {("L", "R")})
    C = L.contract(R, {("R", "L")})
    return {"C": C, "AL": AL, "AR": AR}


def prepare_h(name):
    H = Tensor(["I1", "I2", "O1", "O2"], [2, 2, 2, 2]).zero()
    block = H.block[{}]
    if name == "Ising":
        block[0, 0, 0, 0] = 1
        block[0, 1, 0, 1] = 1
        block[1, 0, 1, 0] = -1
        block[1, 1, 1, 1] = -1
        block[1, 1, 0, 0] = 1
        block[1, 0, 0, 1] = 1
        block[0, 1, 1, 0] = 1
        block[0, 0, 1, 1] = 1
    else:
        raise Exception("Unknown Hamiltonian")
    return H


def prepare_H(name):
    identity = np.array([[1, 0], [0, 1]])
    sigmax = np.array([[0, 1], [1, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    if name == "Ising":
        "H = sigma x sigma x + sigma z"
        """
        (R) ---- I ----> (R)
        (R) - sigma x -> (I) 
        (R) - sigma z -> (F)
        (I) - sigma x -> (F)
        (F) ---- U ----> (F)
        """
        res = Tensor(["L", "R", "I", "O"], [3, 3, 2, 2]).zero()
        block = res.block[["L", "R", "I", "O"]]
        block[0, 0] = identity  # R -> R
        block[0, 1] = sigmax  # R -> I
        block[0, 2] = sigmaz  # R -> F
        block[1, 2] = sigmax  # I -> F
        block[2, 2] = identity  # F -> F
        return res
    else:
        raise Exception("Unknown Hamiltonian")


def get_dimension(tensor, name):
    return tensor.edge[tensor.name.index(name)]


def get_L_or_R(*, AL, H, l, r):
    D = get_dimension(AL, l)
    DH = get_dimension(H, l)
    rh = r + "H"
    rp = r + "'"
    res = Tensor([r, rh, rp], [D, DH, D]).randn()
    for t in range(1000):
        res = res.contract(AL, {(r, l)})
        res = res.contract(H.edge_rename({r: rh}), {(rh, l), ("P", "I")})
        res = res.contract(AL.edge_rename({r: rp}), {(rp, l), ("O", "P")})
        res /= res.norm_max()
    return res


def get_HC_and_HAC(*, H, AL, AR):
    L = get_L_or_R(AL=AL, H=H, l="L", r="R")
    R = get_L_or_R(AL=AR, H=H, l="R", r="L")
    HC = L.contract(R, {("RH", "LH")})
    HAC = L.contract(H.edge_rename({
        "R": "RH",
        "I": "P",
        "O": "P'"
    }), {("RH", "L")}).contract(R, {("RH", "LH")})
    return HC, HAC


def get_ground_state(matrix):
    names = [name for name in matrix.name if str(name)[-1] != "'"]
    dimensions = [get_dimension(matrix, name) for name in names]
    res = Tensor(names, dimensions).randn()
    names_pair = {(name, str(name) + "'") for name in names}
    operator = (-1 * matrix).exponential(names_pair, step=8)
    for t in range(1000):
        res = res.contract(matrix, names_pair)
        res /= res.norm_max()
    return res


def get_new_AL_and_AR(*, C, AC):
    CD = C.edge_rename({"R": "L", "L": "R"})
    ACCD = AC.contract(CD, {("R", "L")})
    CDAC = CD.contract(AC, {("R", "L")})
    UL, _, VL = ACCD.svd({"L", "P"}, "R", "L")
    AL = UL.contract(VL, {("R", "L")})
    UR, _, VR = CDAC.svd({"L"}, "R", "L")
    AR = UR.contract(VR, {("R", "L")})
    return AL, AR


def get_hL_or_hR(AL, h, l, r):
    AA = AL.edge_rename({
        "P": "P1"
    }).contract(AL.edge_rename({"P": "P2"}), {(r, l)})
    HAA = AA.contract(h, {("P1", "I1"), ("P2", "I2")})
    AHA = HAA.contract(AA.edge_rename({r: r + "'"}), {("O1", "P1"),
                                                      ("O2", "P2"), (l, l)})
    return AHA


def get_HL_or_HR(AL, h, l, r):
    hL = get_hL_or_hR(AL, h, l, r)
    res = hL
    for i in range(1000):
        res = res.contract(AL, {(r, l)}).contract(AL.edge_rename({r: r + "'"}),
                                                  {("P", "P"), (r + "'", l)})
        res = res + hL
        res /= res.norm_max()
    return res


def applyHAC(AL, AC, AR, h, HL, HR):
    T1 = AL.edge_rename({
        "P": "P1"
    }).contract(AC.edge_rename({"P": "P2"}), {("R", "L")}).contract(
        h, {("P1", "I1"),
            ("P2", "I2")}).contract(AL.edge_rename({
                "R": "L",
                "L": "L'"
            }), {("L", "L'"), ("O1", "P")}).edge_rename({"O2": "P"})
    T2 = AC.edge_rename({
        "P": "P1"
    }).contract(AR.edge_rename({"P": "P2"}), {("R", "L")}).contract(
        h, {("P1", "I1"),
            ("P2", "I2")}).contract(AR.edge_rename({
                "L": "R",
                "R": "R'"
            }), {("R", "R'"), ("O2", "P")}).edge_rename({"O1": "P"})
    T3 = AC.contract(HL, {("L", "R")}).edge_rename({"R'": "L", "L": "R"})
    T4 = AC.contract(HR, {("R", "L")}).edge_rename({"L'": "R", "R": "L"})
    return T1 + T2 + T3 + T4


def applyHC(AL, C, AR, h, HL, HR):
    T1 = AL.edge_rename({
        "P": "P1"
    }).contract(C, {("R", "L")}).contract(
        AR.edge_rename({"P": "P2"}), {("R", "L")}).contract(
            h,
            {("P1", "I1"),
             ("P2", "I2")}).contract(AL.edge_rename({
                 "L": "L'",
                 "R": "L"
             }), {("L", "L'"),
                  ("O1", "P")}).contract(AR.edge_rename({
                      "R": "R'",
                      "L": "R"
                  }), {("R", "R'"), ("O2", "P")})
    T2 = C.contract(HL, {("L", "R")}).edge_rename({"R'": "L"})
    T3 = C.contract(HR, {("R", "L")}).edge_rename({"L'": "R"})
    return T1 + T2 + T3


def check_energy(*, AL, AR, C):
    H = Tensor(["I1", "I2", "O1", "O2"], [2, 2, 2, 2]).zero()
    block = H.block[{}]
    block[0, 0, 0, 0] = 1
    block[0, 1, 0, 1] = 1
    block[1, 0, 1, 0] = -1
    block[1, 1, 1, 1] = -1
    block[1, 1, 0, 0] = 1
    block[1, 0, 0, 1] = 1
    block[0, 1, 1, 0] = 1
    block[0, 0, 1, 1] = 1
    ###
    psi = AL.edge_rename({
        "P": "PL"
    }).contract(C, {("R", "L")}).contract(AR.edge_rename({"P": "PR"}),
                                          {("R", "L")})
    Hpsi = psi.contract(H, {("PL", "I1"), ("PR", "I2")}).edge_rename({
        "O1": "PL",
        "O2": "PR"
    })
    psipsi = psi.contract_all_edge(psi)
    psiHpsi = Hpsi.contract_all_edge(psi)
    print(float(psiHpsi / psipsi))


def vumps(*, AL, AR, C, H, h):
    for t in range(1000):
        HL = get_HL_or_HR(AL, h, "L", "R")
        HR = get_HL_or_HR(AR, h, "R", "L")

        HC, HAC = get_HC_and_HAC(H=H, AL=AL, AR=AR)
        AC = get_ground_state(HAC).edge_rename({"R": "L", "L": "R"})
        AC = applyHAC(AL, AC, AR, h, HL, HR)
        C = get_ground_state(HC).edge_rename({"L": "R", "R": "L"})
        C = applyHC(AL, C, AR, h, HL, HR)
        AL, AR = get_new_AL_and_AR(C=C, AC=AC)
        check_energy(AL=AL, C=C, AR=AR)
    return AL, C, AR


vumps(**prepare_input(D=3), H=prepare_H("Ising"), h=prepare_h("Ising"))
