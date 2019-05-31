import numpy as np
from scipy.linalg import logm, sqrtm, expm
import scipy

def main(A, B):
    assert A.shape==(3, 3) and B.shape == (3, 3), 'shape error'
    lambda_a = np.linalg.eigvals(A)
    lambda_b = np.linalg.eigvals(B)
    print(lambda_a, lambda_b)
    lambda_b = set(lambda_b)
    has_only_solve = True
    for lambda_1 in lambda_a:
        if lambda_1 in lambda_b:
            has_only_solve = False

    eye = np.eye(3)
    s = np.kron(eye, A) - np.kron(B.T, eye)
    #print(np.linalg.matrix_rank(s))
    #print(s)
    #return has_only_solve, np.linalg.solve(s, np.zeros(9)).reshape(3, 3)
    return has_only_solve, np.linalg.lstsq(s, np.zeros(9))

def main2(A, B):
    print(A)
    alpha = logm(A)
    print(alpha)
    print(expm(alpha))
    beta = logm(B)
    print(beta)
    print(expm(beta))
    M = np.matmul(beta, alpha.T)
    print(M)
    theta_x = np.matmul(sqrtm(np.matmul(M.T, M)), M.T)
    return theta_x

def matlab_svd(mat):
    m, n = mat.shape[:2]
    U, sdiag, VH = np.linalg.svd(mat)
    #U, sdiag, VH = scipy.linalg.svd(mat, lapack_driver='gesvd')
    S = np.zeros((m, n))
    np.fill_diagonal(S, sdiag)
    V = VH.T.conj()
    return U, S, V

def main3(AA, BB):
    np.set_printoptions(precision=6)
    m, n = AA.shape[:2]
    n //= 4
    A = np.zeros((9*n, 9))
    b = np.zeros((9*n, 1))
    eye = np.eye(3)
    for i in range(n):
        Ra = AA[:3, 4*i:4*i+3]
        Rb = BB[:3, 4*i:4*i+3]
        A[9*i:9*i+9, :] = np.kron(Ra, eye) - np.kron(eye, Rb.T)
    u, s, v = matlab_svd(A)
    x = v[:, -1]
    R = x.reshape(3, 3)
    R = np.sign(np.linalg.det(R))/pow(abs(np.linalg.det(R)), 1/3) * R
    u, s, v = matlab_svd(R)
    R = np.matmul(u, v.T)
    if np.linalg.det(R) < 0:
        R = np.matmul(u, np.diag([1, 1, -1]), v.T)
    C = np.zeros((3*n, 3))
    d = np.zeros((3*n, 1))
    for i in range(n):
        C[3*i:3*i+3] = eye - AA[:3, 4*i:4*i+3]
        d[3*i:3*i+3] = (AA[:3, 4*i+3] - np.matmul(R, BB[:3, 4*i+3])).reshape(3, 1)
    #t = np.linalg.solve(C, d)
    t = np.linalg.lstsq(C, d)[0]
    return np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])


if __name__ == '__main__':
    A = np.array([-0.989992, -0.141120, 0, 0, 0.070737, 0, 0.997495, -400,
        0.141120, -0.989992, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, -0.997495, 0, 0.070737, 400,
        0, 0, 0, 1, 0, 0, 0, 1]).reshape(4, 8)
    B = np.array([-0.989992, -0.138307, 0.028036, -26.9559, 0.070737, 0.198172, 0.997612, -309.543,
        0.138307, -0.911449, 0.387470, -96.1332, -0.198172, 0.963323, -0.180936, 59.0244,
        -0.028036, 0.387470, 0.921456, 19.4872,  -0.977612, -0.180936, 0.107415, 291.177,
        0, 0, 0, 1, 0, 0, 0, 1]).reshape(4,8)
    a = main3(A, B)
    ###
    #[[ 9.999995e-01  9.722669e-04 -1.975720e-04  1.003375e+01]
    # [-9.921375e-04  9.801331e-01 -1.983386e-01  4.999007e+01]
    # [ 8.088249e-07  1.983387e-01  9.801335e-01  9.999999e+01]
    # [ 0.000000e+00  0.000000e+00  0.000000e+00  1.000000e+00]]
    print(a)
