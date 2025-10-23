
import numpy as np

def hf_scf_h2():
    """
    一个简单的Hartree-Fock SCF程序，用于计算H2分子在STO-3G基组下的基态能量。
    假设所有单电子和双电子积分都是已知的。
    键长 R = 1.4 a.u.
    """
    # --- 1. 初始化：定义常量和已知积分 ---

    # 原子核排斥能
    E_nuc = 0.7142857143

    # 基函数数量
    n_basis = 2
    # 电子数量
    n_electrons = 2
    # 占据轨道数
    n_occ = n_electrons // 2

    # 重叠积分矩阵 (S)
    S = np.array([
        [1.0000, 0.6593],
        [0.6593, 1.0000]
    ])

    H_core = np.array([
        [-1.1204, -0.9584],
        [-0.9584, -1.1204]
    ])

    def fill_eri(i, j, k, l, value):
        """利用对称性填充双电子积分"""
        indices = [
            (i,j,k,l), (j,i,k,l), (i,j,l,k), (j,i,l,k),
            (k,l,i,j), (l,k,i,j), (k,l,j,i), (l,k,j,i)
        ]
        for idx in indices:
            ERI[idx] = value

    # 双电子排斥积分 (Electron Repulsion Integrals - ERI)
    # ERI[i, j, k, l] = (ij|kl)
    ERI = np.zeros((n_basis, n_basis, n_basis, n_basis))
    fill_eri(0, 0, 0, 0, 0.7746)  # (11|11)
    fill_eri(0, 0, 0, 1, 0.4441)  # (11|12)
    fill_eri(0, 0, 1, 1, 0.5697)  # (11|22)
    fill_eri(0, 1, 0, 1, 0.2970)  # (12|12)
    fill_eri(0, 1, 1, 1, 0.4441)  # (12|22)
    fill_eri(1, 1, 1, 1, 0.7746)  # (22|22)

    # --- 2. 构建正交化矩阵 S^(-1/2) ---
    s_val, s_vec = np.linalg.eigh(S)
    S_inv_sqrt = s_vec @ np.diag(s_val**(-0.5)) @ s_vec.T

    # --- 3. 初始猜测：使用核心哈密顿量 ---
    F_prime = S_inv_sqrt.T @ H_core @ S_inv_sqrt
    _, C_prime = np.linalg.eigh(F_prime)
    C = S_inv_sqrt @ C_prime
    
    # 从占据轨道构建密度矩阵 P
    C_occ = C[:, :n_occ]
    P = 2 * (C_occ @ C_occ.T)

    # --- 4. SCF 迭代循环 ---
    max_iter = 50
    conv_threshold = 1e-6
    E_total = 0.0

    print("SCF 迭代开始:")
    print("Iter | Energy (a.u.)      | Delta E")
    print("-----------------------------------------")

    for i in range(max_iter):
        E_old = E_total

        # a. 构建 Fock 矩阵 F = H_core + G
        G = np.zeros((n_basis, n_basis))
        for mu in range(n_basis):
            for nu in range(n_basis):
                for lam in range(n_basis):
                    for sig in range(n_basis):
                        # G_mu_nu = sum_ls P_ls * [(mu nu|ls) - 0.5 * (mu ls|nu)]
                        coulomb = ERI[mu, nu, lam, sig]
                        exchange = ERI[mu, lam, nu, sig]
                        G[mu, nu] += P[lam, sig] * (coulomb - 0.5 * exchange)
        
        F = H_core + G

        # b. 求解 Roothaan-Hall 方程 FC=SCE
        F_prime = S_inv_sqrt.T @ F @ S_inv_sqrt
        E_orb, C_prime = np.linalg.eigh(F_prime)
        C = S_inv_sqrt @ C_prime

        # c. 构建新的密度矩阵
        C_occ = C[:, :n_occ]
        P = 2 * (C_occ @ C_occ.T)

        # d. 计算总能量
        E_electronic = 0.5 * np.sum(P * (H_core + F))
        E_total = E_electronic + E_nuc
        
        delta_E = E_total - E_old
        
        print(f"{i+1:<4} | {E_total:<18.12f} | {delta_E:9.2e}")

        # e. 检查收敛性
        if abs(delta_E) < conv_threshold:
            print("\nSCF 收敛成功！")
            break
    else:
        print("\nSCF 未能在指定迭代次数内收敛。")

    print("\n--- 最终结果 ---")
    print(f"迭代次数: {i+1}")
    print(f"总能量: {E_total:.12f} a.u.")
    print(f"轨道能: {E_orb}")

if __name__ == "__main__":
    hf_scf_h2()
