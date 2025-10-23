# H2 Molecule Hartree-Fock SCF Calculation

This project contains a simple Python script to calculate the ground state energy of the H2 molecule using the Hartree-Fock Self-Consistent Field (SCF) method in a minimal STO-3G basis.

## Prerequisites

*   Python 3
*   NumPy

You can install NumPy using pip:

```bash
pip install numpy
```

## How to Run

To run the calculation, simply execute the following command in your terminal:

```bash
python hf_scf_h2.py
```

## Example Output

The script will print the progress of the SCF iterations and the final results.

```
SCF 迭代开始:
Iter | Energy (a.u.)      | Delta E
-----------------------------------------
1    | -1.116752940304    | -1.12e+00
2    | -1.116752940304    |  0.00e+00

SCF 收敛成功！

--- 最终结果 ---
迭代次数: 2
总能量: -1.116752940304 a.u.
轨道能: [-0.5782212   0.67048936]
```
