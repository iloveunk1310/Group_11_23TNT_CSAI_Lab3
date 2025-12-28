import numpy as np

def print_info():
    print("Bài toán: Tìm giá trị nhỏ nhất của hàm mục tiêu f với các ràng buộc biến")
    print('Các biến chính dùng trong quy hoạch tuyến tính:')
    print('x: Biến quyết định - vector n chiều - là đại lượng cần tìm')
    print('c: vector hệ số hàm mục tiêu - vector n chiều - là mức “chi phí”, “trọng số”, “lợi ích” của từng biến quyết định')
    print('A: ma trận ràng buộc - ma trận m x n - mô tả cấu trúc tuyến tính của các ràng buộc, mỗi hàng là 1 ràng buộc, mỗi cột là một biến quyết định')
    print('b: vector vế phải - vector m chiều - là hằng số giới hạn "tài nguyên"')
    print('Biến phụ:')
    print('s: biến slack : Ax <= b -> Ax + s = b, s >= 0')
    print('x+, x-: biến tách dấu : x = (x+) - (x-), với x+, x- >= 0')
    print("Giả sử bài toán có n biến số và m ràng buộc đẳng thức (m <= n), ta gọi:")
    print('B: tập biến cơ sở - có m phần tử, tất cả các giá trị này đều không âm')
    print('V: tập biến ngoài cơ sở - tất cả các biến trong này đều có giá trị bằng 0')
    print('A_B, A_V: tương ứng là các ma trận cơ sở và ngoài cơ sở')

class LinearProgram:
    def __init__(self, A, b, c):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)


def get_vertex(B, lp: LinearProgram):
    """
    Xác định tọa độ cụ thể của một đỉnh dựa trên một tập chỉ số cơ sở B cho trước\n
    B: Tập biến cơ sở\n
    lp: bài toán quy hoạch tuyến tính ở dạng đẳng thức
    """
    A, b, c = lp.A, lp.b, lp.c
    n = len(c)

    B = sorted(B)

    # Trích xuất ma trận cơ sở A_B, ma trận này có kích thước m x m
    A_B = A[:, B]

    # Giải hệ phương trình, tính các giá trị của tất cả các biến nằm trong tập biến cơ sở B
    x_B = np.linalg.solve(A_B, b)

    # Khởi tạo vector x toàn số 0, sau đó điền các giá trị x_B vào đúng vị trí của chúng theo chỉ số B
    x = np.zeros(n)
    x[B] = x_B
    return x

def edge_transition(lp: LinearProgram, B, q):
    """
    Xác định xem biến nào sẽ rời khỏi tập cơ sở để nhường chỗ cho biến mới đi vào
    \nq: biến ngoài cơ sở chuẩn bị được thay vào B
    \nThực hiện kiểm tra tỷ lệ tối thiểu (Minimum Ratio Test)
    """
    A, b = lp.A, lp.b
    n = A.shape[1]

    B = sorted(B)
    V = [i for i in range(n) if i not in B]

    A_B = A[:, B]
    A_q = A[:, V[q]]

    # Tính hướng di chuyển d, giải hệ để biết khi tăng biến ngoài cơ sở q thì các biến cơ sở hiện tại sẽ thay đổi như thế nào
    d = np.linalg.solve(A_B, A_q)
    x_B = np.linalg.solve(A_B, b)

    best_ratio = np.inf
    leaving_index = None

    # Duyệt qua các thành phần của d
    # ratio là khoảng cách có thể đi trước khi một biến cơ sở chạm mức 0
    for i in range(len(d)):
        if d[i] > 0:
            ratio = x_B[i] / d[i]
            if ratio < best_ratio:
                best_ratio = ratio
                leaving_index = i

    return leaving_index, best_ratio

def simplex_step(lp: LinearProgram, B, tol=1e-9):
    """
    Tìm kiếm đỉnh tốt nhất để di chuyển tới
    """
    A, b, c = lp.A, lp.b, lp.c
    n = len(c)

    B = sorted(B)
    V = [i for i in range(n) if i not in B]

    A_B = A[:, B]
    A_V = A[:, V]

    x_B = np.linalg.solve(A_B, b)
    c_B = c[B]

    # Tính nhân tử Lagrange lam (lambda)
    lam = np.linalg.solve(A_B.T, c_B)

    # Tính reduced cost mu_V: xác định xem đưa các biến ngoài cơ sở có làm giảm hàm mục tiêu không
    mu_V = c[V] - A_V.T @ lam

    # Chọn biến vào q có mu (mu_q) < 0 giúp làm giảm hàm mục tiêu nhiều nhất
    entering = None
    for i, mu in enumerate(mu_V):
        if mu < -tol:
            entering = i
            break

    if entering is None:
        return B, True  # đã tối ưu

    # Chọn biến đi ra khỏi tập cơ sở
    p, step = edge_transition(lp, B, entering)
    if step == np.inf:
        raise ValueError("Unbounded LP")

    # Cập nhật tập cơ sở mới, sắp xếp lại các chỉ số trong B
    B[p] = V[entering]
    return sorted(B), False

def simplex_optimize(lp: LinearProgram, B):
    """
    Lặp lại thuật toán simplex đến khi tối ưu
    """
    optimal = False
    while not optimal:
        B, optimal = simplex_step(lp, B)
    return B

def find_partition(lp: LinearProgram):
    """
    Tạo bài toán phụ (Auxiliary LP) trong trường hợp chưa biết tập cơ sở B:
    """
    A, b = lp.A, lp.b
    m, n = A.shape
    
    # Xây dựng bài toán phụ
    # Tạo ma trận đường chéo Z: Z_ii = 1 nếu b_i >= 0, ngược lại -1. Dùng để đảm bảo nghiệm khởi tạo z = |b| >= 0
    diag_values = np.where(b >= 0, 1.0, -1.0)
    Z = np.diag(diag_values)
    
    # Ma trận ràng buộc mở rộng A' = [A, Z]
    A_prime = np.hstack([A, Z])
    
    # Hàm mục tiêu phụ c': các biến x hệ số 0, các biến z hệ số 1.
    # Mục tiêu là tổng các phần tử trong z phải về 0
    c_prime = np.concatenate([np.zeros(n), np.ones(m)])
    
    lp_init = LinearProgram(A_prime, b, c_prime)
    
    # Tập cơ sở khởi đầu cho bài toán phụ
    # Chọn tất cả các biến z (chỉ số từ n đến n+m-1)
    B_init = list(range(n, n + m))
    
    # Giải bài toán phụ
    B_temp = simplex_optimize(lp_init, B_init)

    # Kiểm tra tính khả thi
    # Nếu giá trị hàm mục tiêu phụ > 0 (với sai số nhỏ), bài toán gốc vô nghiệm
    x_init = get_vertex(B_temp, lp_init)
    if np.dot(c_prime, x_init) > 1e-6:
        raise ValueError("Bài toán vô nghiệm")

    # Loại bỏ các biến z còn sót lại
    # B_temp có thể chứa chỉ số >= n (biến z), cần thay bằng biến x (chỉ số < n).
    
    # Các biến x đang không nằm trong B
    available_xs = [i for i in range(n) if i not in B_temp]
    
    # Các vị trí trong B đang chứa biến z
    indices_with_z = [i for i, idx in enumerate(B_temp) if idx >= n]
    
    # Lấy các biến x chưa dùng đắp vào chỗ các biến z thừa
    for i, idx_in_B in enumerate(indices_with_z):
        if i < len(available_xs):
            B_temp[idx_in_B] = available_xs[i]
            
    # Sắp xếp lại các phần tử
    B_final = sorted(B_temp)
    
    # Kiểm tra đảm bảo không còn chỉ số nào >= n (không còn biến z)
    if any(idx >= n for idx in B_final):
        raise ValueError("Không thể loại bỏ các biến giả. Có thể vẫn còn ràng buộc chưa xử lý")
        
    return B_final

def minimize_lp(lp: LinearProgram):
    """
    Thuật toán hoàn chỉnh:
    B1. Tìm tập cơ sở khởi tạo (Pha 1)
    B2. Tối ưu hóa bài toán gốc (Pha 2)
    B3. Trả về kết quả
    """
    # Pha 1: Initialization
    B = find_partition(lp)
    print(f"Tập biến cơ sở: {B}")
    
    # Pha 2: Optimization
    B_opt = simplex_optimize(lp, B)
    
    x_opt = get_vertex(B_opt, lp)
    return x_opt, B_opt


def get_lambda(B, lp: LinearProgram):
    """
    Tính biến đối ngẫu lambda từ tập cơ sở B.
    Công thức: lambda = (A_B)^-T * c_B
    """
    A, c = lp.A, lp.c
    
    # Sắp xếp lại B để khớp với thứ tự cột trong ma trận
    B = sorted(B)
    
    # Lấy ma trận con A_B và vector chi phí c_B
    A_B = A[:, B]
    c_B = c[B]
    
    # Giải hệ phương trình A_B.T * lambda = c_B
    lam = np.linalg.solve(A_B.T, c_B)
    return lam

def dual_certificate(lp: LinearProgram, x, lam, eps=1e-6):
    """
    Kiểm tra lại nghiệm (3 điều kiện)
    """
    A, b, c = lp.A, lp.b, lp.c

    # Kiểm tra Ax = b và x >= 0
    primal_feasible = np.all(x >= -eps) and np.allclose(A @ x, b, atol=eps)
    
    # Kiểm tra A^T * lam <= c
    dual_feasible = np.all(A.T @ lam <= c + eps)

    # Giá trị hàm mục tiêu của bài toán gốc c^T * x phải (xấp xỉ) bằng giá trị của b^T * lam
    strong_duality = abs(c @ x - b @ lam) <= eps

    return primal_feasible and dual_feasible and strong_duality


if __name__ == "__main__":
    print_info()

    # Ví dụ 1:
    # Minimize -x1 - x2
    # Ràng buộc:
    # 2x1 + x2 <= 4  -> 2x1 + x2 + x3 = 4
    # x1 + 2x2 <= 3  -> x1 + 2x2 + x4 = 3
    # x >= 0

    # Uncommand A, b, c dưới đây để test
    c = [-1, -1, 0, 0]
    A = [[2, 1, 1, 0],
         [1, 2, 0, 1]]
    b = [4, 3]
    

    # Ví dụ 2:
    # Minimize 2x1 - 3x2
    # Ràng buộc:
    # x1 + 2x2 >= 5  -> x1 + 2x2 - x3 = 5
    # x2 <= 1  -> x2 + x4 = 1
    # x >= 0

    # Uncommand A, b, c dưới đây để test
    # c = [2, -3, 0, 0]
    # A = [[1, 2, -1, 0],
    #      [0, 1, 0, 1]]
    # b = [5, 1]

    # Ví dụ 3 (Bài toán không chặn (Unbounded LP))
    # Minimize x1 - 5x2
    # Ràng buộc:
    # |2x1 - x2| <= 2, tương đương với:
    # 2x1 - x2 <= 2  -> 2x1 - x2 + x3 = 2
    # 2x1 - x2 >= -2 -> 2x1 - x2 - x4 = -2
    # x >= 0

    # Uncommand A, b, c dưới đây để test
    # c = [1, -5, 0, 0]
    # A = [[2, -1, -1, 0],
    #      [2, -1, 0, 1]]
    # b = [2, -2]

    # Ví dụ 4 (Mâu thuẫn - Bài toán vô nghiệm)
    # Minimize x1 - 5x2
    # Ràng buộc:
    # 2x1 - x2 >= 2  -> 2x1 - x2 - x3 = 2
    # 2x1 - x2 <= -2 -> 2x1 - x2 + x4 = -2
    # x >= 0

    # Uncommand A, b, c dưới đây để test
    # c = [1, -5, 0, 0]
    # A = [[2, -1, -1, 0],
    #      [2, -1, 0, 1]]
    # b = [2, -2]

    lp = LinearProgram(A, b, c)
    
    try:
        x_opt, B_opt = minimize_lp(lp)
        print("Basis tối ưu (B):", B_opt)
        print("Nghiệm (x):", x_opt)
        print("Giá trị hàm mục tiêu (Primal):", np.dot(c, x_opt))
        
        lam_opt = get_lambda(B_opt, lp)
        print("Biến đối ngẫu (lambda):", lam_opt)
        print("Giá trị hàm mục tiêu đối ngẫu (Dual):", np.dot(b, lam_opt))

        if dual_certificate(lp, x_opt, lam_opt) == 1:
            print('Kết quả đúng')
        else:
            print('Kết quả sai')
            
    except Exception as e:
        print("Lỗi:", e)