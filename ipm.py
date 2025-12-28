import numpy as np
from scipy.optimize import linprog


def solve_ipm(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    Giải bài toán Linear Programming bằng IPM

    min   c^T x
    s.t.  A_ub x <= b_ub
          A_eq x  = b_eq
          b[i][0] <= x_i <= b[i][1]
    """

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='interior-point',
        options={
            'tol': 1e-9,
            'maxiter': 1000,
            'disp': False
        }
    )

    return res

if __name__ == "__main__":
    
    # Ví dụ minh họa:
    #     maximize  3 x1 + 3 x2
    #     subject to x1 + x2 <= 4
    #                x1, x2 >= 0
    
    #bài toán (đổi ngược thành minimize)
    c = np.array([-3.0, -3.0])

    # Bất đẳng thức: x1 + x2 <= 4
    A_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([4.0])

    # Không có ràng buộc đẳng thức
    A_eq = None
    b_eq = None

    # Biên: x1, x2 >= 0
    bounds = [(0, None), (0, None)]

    # Giải
    result = solve_ipm(c, A_ub, b_ub, A_eq, b_eq, bounds)

    # In kết quả
    print("=== Interior-Point Method (bài 1))")
    if result.success:
        print("Status: Giải thành công")
        print("Nghiệm tối ưu x:", result.x)
        print("GIá trị hàm mục tiêu:", -result.fun)  #đổi dấu
    else:
        print("Status: Giải thất bại")
        print("Lỗi:", result.message)

    #Ví dụ 2
    # Minimize -x1 - x2
    # Ràng buộc:
    # 2x1 + x2 <= 4  
    # x1 + 2x2 <= 3  
    # x >= 0
    print("=== Interior-Point Method (bài 2))")
    c = np.array([-1.0, -1.0])
    A_ub = np.array([[2.0, 1.0], [1.0, 2.0]])
    b_ub = np.array([4.0, 3.0])
    A_eq = None
    b_eq = None
    bounds = [(0, None), (0, None)]
    result = solve_ipm(c, A_ub, b_ub, A_eq, b_eq, bounds)
    if result.success:
        print("Status: Giải thành công")
        print("Nghiệm tối ưu x:", result.x)
        print("GIá trị hàm mục tiêu:", result.fun)
    else:
        print("Status: Giải thất bại")
        print("Lỗi:", result.message)




