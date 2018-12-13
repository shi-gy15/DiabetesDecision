from cvxopt import matrix, normal, blas, setseed, mul, lapack, solvers
from cvxopt import exp, div, sqrt
import numpy as np


def mcsvm(X, labels, gamma, kernel = 'linear', sigma = 1.0, degree = 1):
    N, n = X.size
    m = max(labels) + 1
    E = matrix(0.0, (N, m))
    E[matrix(range(N)) + N * matrix(labels)] = 1.0

    def G(x, y, alpha=1.0, beta=0.0, trans='N'):
        blas.scal(beta, y) 
        blas.axpy(x, y, alpha)

    h = matrix(gamma*E, (N*m, 1))
    ones = matrix(1.0, (m,1))

    def A(x, y, alpha=1.0, beta=0.0, trans='N'):
        if trans == 'N': 
            blas.gemv(x, ones, y, alpha=alpha, beta=beta)
        else: 
            blas.scal(beta, y)
            blas.ger(x, ones, y, alpha=alpha)
        
    b = matrix(0.0, (N,1))

    if kernel == 'linear' and N > n:
        def P(x, y, alpha=1.0, beta=0.0):
            z = matrix(0.0, (n, m))
            blas.gemm(X, x, z, transA='T')
            blas.gemm(X, z, y, alpha=alpha, beta=beta)
    else:
        if kernel == 'linear':
            Q = matrix(0.0, (N, N))
            blas.syrk(X, Q)
        elif kernel == 'poly':
            Q = matrix(0.0, (N,N))
            blas.syrk(X, Q, alpha=1.0 / sigma)
            Q = Q**degree
        else:
            raise ValueError("invalid kernel type")

        def P(x, y, alpha=1.0, beta=0.0):
            blas.symm(Q, x, y, alpha=alpha, beta=beta)

    if kernel == 'linear' and N > n:  # add separate code for n <= N <= m*n

        H = [ matrix(0.0, (n, n)) for k in range(m) ]
        S = matrix(0.0, (m*n, m*n))
        Xs = matrix(0.0, (N, n))
        wnm = matrix(0.0, (m*n, 1))
        wN = matrix(0.0, (N, 1))
        D = matrix(0.0, (N, 1))

        def kkt(W):
            d = matrix(W['d'], (N, m))
            dsq = matrix(W['d']**2, (N, m))

            for k in range(m):
                blas.scal(0.0, H[k])
                H[k][::n+1] = 1.0
                blas.copy(X, Xs)
                for j in range(n):
                    blas.tbmv(d, Xs, n=N, k=0, ldA=1, offsetA=k*N, offsetx=j*N)
                blas.syrk(Xs, H[k], trans='T', beta=1.0)

                lapack.potrf(H[k])

            blas.gemv(dsq, ones, D)
            D[:]=sqrt(D)
            blas.scal(0.0, S)
            for i in range(m):
                for j in range(i+1):
                    blas.copy(X, Xs)
                    blas.copy(d, wN, n=N, offsetx=i*N)
                    blas.tbmv(d, wN, n=N, k=0, ldA=1, offsetA=j*N)
                    blas.tbsv(D, wN, n=N, k=0, ldA=1)
                    for k in range(n):
                        blas.tbmv(wN, Xs, n=N, k=0, ldA=1, offsetx=k*N)
                    blas.gemm(Xs, Xs, S, transA='T', ldC=m*n, offsetC=(j*n)*m*n + i*n)

            for i in range(m):
                blas.trsm(H[i], S, m=n, n=(i+1)*n, ldB=m*n, offsetB=i*n)
                blas.trsm(H[i], S, side='R', transA='T', m=(m-i)*n, n=n, ldB=m*n, offsetB=i*n*(m*n + 1))

            blas.scal(-1.0, S)
            S[::(m*n+1)] += 1.0
            lapack.potrf(S)

            def f(x, y, z):
                blas.tbmv(dsq, x, n=N*m, k=0, ldA=1)
                blas.axpy(z, x)
                blas.gemv(x, ones, y, alpha=1.0, beta=-1.0)
                blas.gemm(X, x, wnm, m=n, k= N, n=m, transA='T', ldB=N, ldC=n)
                for k in range(m):
                    lapack.potrs(H[k], wnm, offsetB=k*n)
                for k in range(m):
                    blas.gemv(X, wnm, wN, offsetx=n*k)
                    blas.tbmv(dsq[:,k], wN, n=N, k=0, ldA=1)
                    blas.axpy(wN, y, -1.0)

                blas.tbsv(D, y, n=N, k=0, ldA=1)
                for k in range(m):
                    blas.copy(y, wN)
                    blas.tbmv(dsq, wN, n=N, k=0, ldA=1, offsetA=k*N)
                    blas.tbsv(D, wN, n=N, k=0, ldA=1)
                    blas.gemv(X, wN, wnm, trans='T', offsety=k*n)
                    blas.trsv(H[k], wnm, offsetx=k*n)

                lapack.potrs(S, wnm)

                for k in range(m):
                    blas.trsv(H[k], wnm, trans='T', offsetx=k*n)
                    blas.gemv(X, wnm, wN, offsetx=k*n)
                    blas.tbmv(dsq, wN, n=N, k=0, ldA=1, offsetA=k*N)
                    blas.tbsv(D, wN, n=N, k=0, ldA=1)
                    blas.axpy(wN, y)

                blas.tbsv(D, y, n=N, k=0, ldA=1)

                for k in range(m):
                    blas.copy(y, wN)
                    blas.tbmv(dsq, wN, n=N, k=0, ldA=1, offsetA=k*N)
                    blas.axpy(wN, x, -1.0, offsety=k*N)

                blas.gemm(X, x, wnm, transA='T', m=n, n=m, k=N, ldB=N, ldC=n)

                for k in range(m):
                    lapack.potrs(H[k], wnm, offsetB=n*k)

                for k in range(m):
                    blas.gemv(X, wnm, wN, offsetx=k*n)
                    blas.tbmv(dsq, wN, n=N, k=0, ldA=1, offsetA=k*N)
                    blas.axpy(wN, x, -1.0, n=N, offsety=k*N)

                blas.axpy(x, z, -1.0)
                blas.scal(-1.0, z)
                blas.tbsv(d, z, n=N*m, k=0, ldA=1)
            return f
    else:
        H = [matrix(0.0, (N, N)) for k in range(m)]
        S = matrix(0.0, (N, N))

        def kkt(W):
            D = matrix(W['di']**2, (N, m))
            blas.scal(0.0, S)
            for k in range(m):
                blas.copy(Q, H[k])
                H[k][::N+1] += D[:, k]
                lapack.potrf(H[k])
                lapack.potri(H[k])
                blas.axpy(H[k], S)
            lapack.potrf(S)

            def f(x, y, z):
                z.size = N, m
                x += mul(D, z)
                for k in range(m):
                    blas.symv(H[k], x[:, k], y, alpha=-1.0, beta=1.0)
                lapack.potrs(S, y)
                w = matrix(0.0, (N, 1))
                for k in range(m):
                    blas.axpy(y, x, offsety=N*k, n=N)
                    blas.symv(H[k], x, w, offsetx=N*k)
                    blas.copy(w, x, offsety=N*k)

                blas.scal(-1.0, y)
                blas.axpy(x, z, -1.0)
                blas.tbsv(W['d'], z, n=m * N, k=0, ldA=1)
                blas.scal(-1.0, z)
                z.size = N * m, 1
            return f

    solvers.options['refinement'] = 1
    sol = solvers.coneqp(P, -E, G, h, A=A, b=b, kktsolver=kkt, xnewcopy=matrix, xdot=blas.dot, xaxpy=blas.axpy, xscal=blas.scal)
    U = sol['x']

    if kernel == 'linear':
        W = matrix(0.0, (n, m))
        blas.gemm(X, U, W, transA='T')
        def classifier(Y):
            M = Y.size[0]
            S = Y * W
            c = []
            for i in range(M):
               a = list(zip(list(S[i, :]), range(m)))
               a.sort(reverse = True)
               c += [ a[0][1] ]
            return c
    elif kernel == 'poly':
        def classifier(Y):
            M = Y.size[0]
            K = matrix(0.0, (M, N))
            blas.gemm(Y, X, K, transB = 'T', alpha = 1.0/sigma)
            S = K**degree * U
            c = []
            for i in range(M):
               a = list(zip(list(S[i,:]), range(m)))
               a.sort(reverse = True)
               c += [ a[0][1] ]
            return c
    else:
        pass

    return classifier
