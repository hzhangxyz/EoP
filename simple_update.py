import TAT
from hamiltonian import get_H
import numpy as np
import sys
from eop import get_eop
from dense_matrix import get_dense_matrix, get_merged_rho

Tensor = TAT(float)


class MPS():
    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, v):
        self._delta_t = v
        self.U = (-self._delta_t * self.H).exponential({("I1", "O1"),
                                                        ("I2", "O2")})

    def __init__(self, D, H, delta_t):
        self.D = D
        self.H = H
        self.delta_t = delta_t

        self.A = Tensor(["P", "L", "R"], [2, D, D]).randn()
        self.B = Tensor(["P", "L", "R"], [2, D, D]).randn()
        self.EAB = None
        self.EBA = None

    def simple_update_single_step(self):
        A = self.A
        B = self.B
        EAB = self.EAB
        EBA = self.EBA
        U = self.U
        D = self.D
        for i in range(2):
            if i == 0:
                this_A = A
                this_B = B
                this_EAB = EAB
                this_EBA = EBA
            else:
                this_A = B
                this_B = A
                this_EAB = EBA
                this_EBA = EAB
            big = this_A
            big = big.edge_rename({"P": "PA"})
            if this_EAB:
                big = big.multiple(this_EAB, "R", 'u')
            big = big.contract(this_B, {("R", "L")})
            big = big.edge_rename({"P": "PB"})
            if this_EBA:
                big = big.multiple(this_EBA, "L", 'v')
                big = big.multiple(this_EBA, "R", 'u')
            big = big.contract(U, {("PA", "I1"), ("PB", "I2")})
            u, s, v = big.svd({"L", "O1"}, "R", "L", D)
            this_EAB = s / s.norm_max()
            this_A = u.edge_rename({"O1": "P"})
            this_B = v.edge_rename({"O2": "P"})
            if this_EBA:
                this_A = this_A.multiple(this_EBA, "L", 'v', True)
                this_B = this_B.multiple(this_EBA, "R", 'u', True)
            if i == 0:
                A = this_A
                B = this_B
                EAB = this_EAB
                EBA = this_EBA
            else:
                B = this_A
                A = this_B
                EBA = this_EAB
                EAB = this_EBA
            print(-np.log(s.norm_max()) / self.delta_t)
        self.A = A
        self.B = B
        self.EAB = EAB
        self.EBA = EBA

    def __call__(self, i):
        if i % 2 == 0:
            return self.A.multiple(self.EAB, "R", "u")
        else:
            return self.B.multiple(self.EBA, "R", "u")


mps = MPS(D=10, H=get_H(), delta_t=0.1)
for i in range(1000):
    mps.simple_update_single_step()
"""
mps.D = 20
for i in range(1000):
    mps.simple_update_single_step()
mps.D = 30
for i in range(1000):
    mps.simple_update_single_step()
delta_t = 0.01
for i in range(1000):
    mps.simple_update_single_step()
delta_t = 0.001
for i in range(1000):
    mps.simple_update_single_step()
delta_t = 0.0001
for i in range(1000):
    mps.simple_update_single_step()
"""

print("STATE DONE")

for i in [j for j in range(1, 101)]:
    rho = get_dense_matrix(2000, [500, 500 + i], mps)
    merged_rho = get_merged_rho(rho, part={0})
    eop = get_eop(merged_rho)
    print(i, eop)
