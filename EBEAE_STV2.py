# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:21:54 2021

@author: alex
"""

# Extended Blind Endmembers and Abundances Extraction (EBEAE) Algorithm
import time
import numpy as np
from scipy.sparse import diags
import scipy.sparse as ssp
from scipy.sparse import linalg
import scipy
# from scipy.linalg import lstsq
# from scipy.optimize import nnls
import concurrent.futures
import multiprocessing as mp
from multiprocessing import freeze_support

def performance(fn):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        # print(f'Function {fn.__name__} took {t2-t1} s')
        return t2 - t1, result
    return wrapper


@performance
def abundance(Y, P, Lambda, parallel):
    """
    A = abundance(Y,P,lambda,parallel):
    Estimation of Optimal Abundances in Linear Mixture Model
    Input Arguments:
    Y = matrix of measurements
    P = matrix of end-members
    Lambda =  entropy weight in abundance estimation in (0,1)
    parallel = implementation in parallel of the estimation
    Output Argument:
    A = abundances matrix
    Daniel U. Campos-Delgado
    September/2020
    """
    # Check arguments dimensions
    numerr = 0
    M, N = Y.shape
    n = P.shape[1]
    A = np.zeros((n, N))
    if P.shape[0] != M:
        print("ERROR: the number of rows in Y and P does not match")
        numerr = 1

    # Compute fixed vectors and matrices
    c = np.ones((n, 1))
    d = 1  # Lagrange Multiplier for equality restriction
    Go = P.T @ P
    w, v = np.linalg.eig(Go)
    l_min = np.amin(w)
    G = Go-np.eye(n)*l_min*Lambda
    while (1/np.linalg.cond(G, 1)) < 1e-6:
        Lambda = Lambda/2
        G = Go-np.eye(n)*l_min*Lambda
        if Lambda < 1e-6:
            print("Unstable numerical results in abundances estimation, update rho!!")
            numerr = 1
    Gi = np.linalg.pinv(G)
    T1 = Gi@c
    T2 = c.T@T1

    # Start Computation of Abundances
    for k in range(N):
        yk = np.c_[Y[:, k]]
        byk = float(yk.T@yk)
        bk = P.T@yk

        # Compute Optimal Unconstrained Solution
        dk = np.divide((bk.T@T1)-1, T2)
        ak = Gi@(bk-c@dk)

        # Check for Negative Elements
        if float(sum(ak >= 0)) != n:
            I_set = np.zeros((1, n))
            while float(sum(ak < 0)) != 0:
                I_set = np.where(ak < 0, 1, I_set.T).reshape(1, n)
                L = len(np.where(I_set == 1)[1])
                Q = n+1+L
                Gamma = np.zeros((Q, Q))
                Beta = np.zeros((Q, 1))
                Gamma[:n, :n] = G/byk
                Gamma[:n, n] = c.T
                Gamma[n, :n] = c.T
                cont = 0
                # if n >= 2:
                #     if bool(I_set[:, 0] != 0):
                #         cont += 1
                #         Gamma[0, n+cont] = 1
                #         Gamma[n+cont, 0] = 1
                #     if bool(I_set[:, 1] != 0):
                #         cont += 1
                #         Gamma[1, n+cont] = 1
                #         Gamma[n+cont, 1] = 1
                #     if n >= 3:
                #         if bool(I_set[:, 2] != 0):
                #             cont += 1
                #             Gamma[2, n+cont] = 1
                #             Gamma[n+cont, 2] = 1
                #         if n == 4:
                #             if bool(I_set[:, 3] != 0):
                #                 cont += 1
                #                 Gamma[3, n+cont] = 1
                #                 Gamma[n+cont, 3] = 1
                for i in range(n):
                    if I_set[:,i] != 0:
                        cont += 1
                        ind = i
                        Gamma[ind, n+cont] = 1
                        Gamma[n+cont, ind] = 1
                        
                Beta[:n, :] = bk/byk
                Beta[n, :] = d
                delta = np.linalg.solve(Gamma, Beta)
                ak = delta[:n]
                ak = np.where(abs(ak) < 1e-9, 0, ak)
        A[:, k] = np.c_[ak].T
    return A, numerr


@performance
def abundance_TV(Y, P, W, parallel, mu):
    """
    A = abundance(Y,P,lambda,parallel):
    Estimation of Optimal Abundances in Linear Mixture Model
    Input Arguments:
    Y = matrix of measurements
    P = matrix of end-members
    Lambda =  entropy weight in abundance estimation in (0,1)
    parallel = implementation in parallel of the estimation
    Output Argument:
    A = abundances matrix
    Daniel U. Campos-Delgado
    September/2020
    """
    # Check arguments dimensions
    numerr = 0
    M, N = Y.shape
    n = P.shape[1]
    A = np.zeros((n, N))
    if P.shape[0] != M:
        print("ERROR: the number of rows in Y and P does not match")
        numerr = 1

    # Compute fixed vectors and matrices
    em = np.eye(n)
    c = np.ones((n, 1))
    # d = 1  # Lagrange Multiplier for equality restriction
    Go = P.T @ P
    # w, v = np.linalg.eig(Go)
    # l_min = np.amin(w)
    # G = Go-np.eye(n)*l_min*Lambda
    # while (1/np.linalg.cond(G, 1)) < 1e-6:
    #     Lambda = Lambda/2
    #     G = Go-np.eye(n)*l_min*Lambda
    #     if Lambda < 1e-6:
    #         print("Unstable numerical results in abundances estimation, update rho!!")
    #         numerr = 1
    # Gi = np.linalg.pinv(G)
    # T1 = Gi@c
    # T2 = c.T@T1

    # Start Computation of Abundances
    # Aux=[]
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for k in range(N):
                # secs=[5,4,3,2,1]
                # results = [executor.submit(do_something,sec) for sec in secs]
                results = executor.submit(ak_estimation,np.c_[Y[:, k]], np.c_[W[:, k]], P, Go, c, em, n, mu)
                A[:,k]=results.result()
                # t_A, outputs_a = abundance_TV(Ym, P, Wn, parallel,sc['mu'])

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     Aux = [executor.map(ak_estimation, (np.c_[Y[:, k]], np.c_[W[:, k]], P, Go, c, em, n, mu)) for k in range(N)]
            
        #     for f in concurrent.futures.as_completed(Aux):
        #         print(f.result())
        # for k in range(N):
        #     Aux=mp.Process(target=ak_estimation, args=(np.c_[Y[:, k]], np.c_[W[:, k]], P, Go, c, em, n, mu))
        #     Aux.start()
        #     processes.append(Aux)
        # for process in processes:
        #     process.join()
            
        #pool = mp.Pool(mp.cpu_count())
        #Aux = pool.starmap(ak_estimation, [(np.c_[Y[:, k]], np.c_[W[:, k]], P, Go, c, em, n, mu) for k in range(N)])
        # Aux = [pool.apply(ak_estimation, args=(np.c_[Y[:, k]], np.c_[W[:, k]], P, Go, c, em, n, mu)) for k in range(N)]
        #pool.close()
        # A = np.concatenate( Aux, axis=0 ).T
        # for k in range(N):
        #     A[:,k]=ak_estimation(np.c_[Y[:, k]], np.c_[W[:, k]], P, Go, c, em, n, mu)
    else:
        for k in range(N):
            A[:,k]=ak_estimation(np.c_[Y[:, k]], np.c_[W[:, k]], P, Go, c, em, n, mu)
            # yk = np.c_[Y[:, k]]
            # w = np.c_[W[:, k]]
            # byk = float(yk.T@yk)
            # bk = P.T@yk + mu*byk*w
            # G = Go + mu*byk*em
            # Gi = em@np.linalg.inv(G)
            # T1 = Gi@c
            # T2 = c.T@T1
    
            # # Compute Optimal Unconstrained Solution
            # dk = (bk.T@T1-1)@np.linalg.inv(T2)
            # ak = Gi@(bk-dk*c)
    
            # # Check for Negative Elements
            # if float(sum(ak >= 0)) != n:
            #     I_set = np.zeros((1, n))
            #     while float(sum(ak < 0)) != 0:
            #         I_set = np.where(ak < 0, 1, I_set.T).reshape(1, n)
            #         L = len(np.where(I_set == 1)[1])
            #         Q = n+1+L
            #         Gamma = np.zeros((Q, Q))
            #         Beta = np.zeros((Q, 1))
            #         Gamma[:n, :n] = G/byk
            #         Gamma[:n, n] = c.T
            #         Gamma[n, :n] = c.T
            #         cont = 0
            #         for i in range(n):
            #             if I_set[:,i] != 0:
            #                 cont += 1
            #                 ind = i
            #                 Gamma[ind, n+cont] = 1
            #                 Gamma[n+cont, ind] = 1
                            
            #         Beta[:n, :] = bk/byk
            #         Beta[n, :] = d
            #         delta = np.linalg.solve(Gamma, Beta)
            #         ak = delta[:n]
            #         ak = np.where(abs(ak) < 1e-9, 0, ak)
            # A[:, k] = np.c_[ak].T
    return A, numerr

def ak_estimation(yk, w, P, Go, c, em, n, mu):
    byk = float(yk.T@yk)
    bk = P.T@yk + mu*byk*w
    G = Go + mu*byk*em
    Gi = em@np.linalg.inv(G)
    T1 = Gi@c
    T2 = c.T@T1

    # Compute Optimal Unconstrained Solution
    dk = (bk.T@T1-1)@np.linalg.inv(T2)
    ak = Gi@(bk-dk*c)

    # Check for Negative Elements
    if float(sum(ak >= 0)) != n:
        I_set = np.zeros((1, n))
        while float(sum(ak < 0)) != 0:
            I_set = np.where(ak < 0, 1, I_set.T).reshape(1, n)
            L = len(np.where(I_set == 1)[1])
            Q = n+1+L
            Gamma = np.zeros((Q, Q))
            Beta = np.zeros((Q, 1))
            Gamma[:n, :n] = G/byk
            Gamma[:n, n] = c.T
            Gamma[n, :n] = c.T
            cont = 0
            for i in range(n):
                if I_set[:,i] != 0:
                    cont += 1
                    ind = i
                    Gamma[ind, n+cont] = 1
                    Gamma[n+cont, ind] = 1
                    
            Beta[:n, :] = bk/byk
            Beta[n, :] = 1
            delta = np.linalg.solve(Gamma, Beta)
            ak = delta[:n]
            ak = np.where(abs(ak) < 1e-9, 0, ak)
    return np.c_[ak].T

@performance
def dn_abundance(A, sc, maxiter, tempval, Dh, Dv):
    
    dim,mn=A.shape
    b1=np.zeros((mn,1));
    W=A.T
    Wr=np.zeros(W.shape)
    b2=b1  
    p=b1   
    q=b1
    # Dh=diags([-1, 1], [0,1],shape=(sc['dimY'],sc['dimY'])).toarray()
    # Dh[sc['dimY']-1,:]=0
    # Dh=ssp.kron(Dh,np.eye(sc['dimX']),format='csr')
    # Dv=diags([-1, 1], [0,1],shape=(sc['dimX'],sc['dimX'])).toarray()
    # Dv[sc['dimX']-1,:]=0
    # Dv=ssp.kron(Dv,np.eye(sc['dimY']),format='csr')
    # tempval= sc['nu']*(Dh.T*Dh+Dv.T*Dv)+ sc['mu']*ssp.eye(mn).toarray()
    # Q,R=np.linalg.qr(tempval)
    for j in range(dim):
        for i in range(maxiter):
            An=np.c_[A[j,:]]
            # Wn=np.c_[W[:,j]]
            Ay=sc['mu']*An + sc['nu']*Dh.T@(p-b1)+sc['nu']*Dv.T@(q-b2)
            # tempval= sc['nu']*((Dh.T*(Dh*W))+(Dv.T*(Dv*W)))+ sc['mu']*W;
            # Wr[:,j], c = np.linalg.lstsq(tempval,Ay)
            # Wn=linalg.solve_triangular(R.asformat(format='array'),Q.T.dot(Ay))
            Wn=ssp.linalg.spsolve(tempval,Ay)
            
            a1=Dh@Wn.reshape([mn,1])
            a2=Dv@Wn.reshape([mn,1])
            
            p=SoftTh(a1+b1,sc['tau']/sc['nu']);
            q=SoftTh(a2+b2,sc['tau']/sc['nu']);
            
            # p=SoftTh(Dh@Wn+b1,sc['tau']/sc['nu']);
            # q=SoftTh(Dv@Wn+b2,sc['tau']/sc['nu']);

            b1=b1+a1-p;
            b2=b2+a2-q;
            
            Wr[:,j]=Wn
    Wr[Wr<0]=0
    return Wr.T
    
    
def SoftTh(B,lamb):
    return np.sign(B)*np.fmax(0,np.abs(B)-lamb*np.ones((len(B),1)))

@performance
def endmember(Y, A, rho, normalization):
    """
    P = endmember(Y,A,rho,normalization)
    Estimation of Optimal End-members in Linear Mixture Model
    Input Arguments:
    Y = Matrix of measurements
    A =  Matrix of abundances
    rho = Weighting factor of regularization term
    normalization = normalization of estimated profiles (0=NO or 1=YES)
    Output Argument:
    P = Matrix of end-members
    Daniel U. Campos-Delgado
    September/2020
    """
    n, N = A.shape
    M, K = Y.shape
    numerr = 0
    R = sum(n-np.array(range(1, n)))
    W = np.tile((1/K/sum(Y**2)), [n, 1]).T
    if Y.shape[1] != N:
        print("ERROR: the number of columns in Y and A does not match")
        numerr = 1
    o = (n * np.eye(n)-np.ones((n, n)))
    n1 = (np.ones((n, 1)))
    m1 = (np.ones((M, 1)))

    # Construct Optimal Endmembers Matrix
    T0 = (A @ (W*A.T)+rho*np.divide(o, R))
    while 1/np.linalg.cond(T0, 1) < 1e-6:
        rho = rho/10
        T0 = (A @ (W*A.T)+rho*np.divide(o, R))
        if rho < 1e-6:
            print("Unstable numerical results in endmembers estimation, update rho!!")
            numerr = 1
    V = (np.eye(n) @ np.linalg.pinv(T0))
    T2 = (Y @ (W*A.T) @ V)
    if normalization == 1:
        T1 = (np.eye(M)-(1/M)*(m1 @ m1.T))
        T3 = ((1/M)*m1 @ n1.T)
        P_est = T1 @ T2 + T3
    else:
        P_est = T2

    # Evaluate and Project Negative Elements
    P_est = np.where(P_est < 0, 0, P_est)
    P_est = np.where(np.isnan(P_est), 0, P_est)
    P_est = np.where(np.isinf(P_est), 0, P_est)

    # Normalize Optimal Solution
    if normalization == 1:
        P_sum = np.sum(P_est, 0)
        P = P_est/np.tile(P_sum, [M, 1])
    else:
        P = P_est
    return P, numerr


@performance
def ebeae_stv(Yo, n, parameters, Po, oae, sc):
    """
    P, A, An, Yh, a_Time, p_Time = ebeae(Yo, n, parameters, Po, oae)
    Estimation of Optimal Endmembers and Abundances in Linear Mixture Model
    Input Arguments:
      Y = matrix of measurements (MxN)
      n = order of linear mixture model
      parameters = 9x1 vector of hyperparameters in EBEAE methodology
                 = [initicond, rho, Lambda, epsilon, maxiter, downsampling, parallel, normalization, display]
          initcond = initialization of endmembers matrix {1,2,3,4}
                                    (1) Maximum cosine difference from mean measurement (default)
                                    (2) Maximum and minimum energy, and largest distance from them
                                    (3) PCA selection + Rectified Linear Unit
                                    (4) ICA selection (FOBI) + Rectified Linear Unit
          rho = regularization weight in endmember estimation (default rho=0.1)
          Lambda = entropy weight in abundance estimation in [0,1) (default Lambda=0)
          epsilon = threshold for convergence in ALS method (default epsilon=1e-3)
          maxiter = maximum number of iterations in ALS method (default maxiter=20)
          downsampling = percentage of random downsampling in endmember estimation [0,1) (default downsampling=0.5)
          parallel = implement parallel computation of abundances (0->NO or 1->YES) (default parallel=0)
          normalization = normalization of estimated end-members (0->NO or 1->YES) (default normalization=1)
          display = show progress of iterative optimization process (0->NO or 1->YES) (default display=0)
      Po = initial end-member matrix (Mxn)
      oae = only optimal abundance estimation with Po (0 -> NO or 1 -> YES) (default oae = 0)
    Output Arguments:
      P  = matrix of endmembers (Mxn)
      A  = scaled abundances matrix (nxN)
      An = abundances matrix normalized (nxN)
      Yh = estimated matrix of measurements (MxN)
      a_Time = estimated time in abundances estimation
      p_Time = estimated time in endmembers estimation
    Daniel U. Campos Delgado
    July/2020
    """
    # Default parameters
    initcond = 1
    rho = 0.1
    Lambda = 0
    epsilon = 1e-3
    maxiter = 20
    downsampling = 0.5
    parallel = 0
    normalization = 1
    display = 0
    numerr = 0
    sc_pass = 1

    # Checking consistency of input arguments
    nargin = 0
    if type(Yo) == np.ndarray:
        nargin += 1
    if type(n) == int:
        nargin += 1
    if type(parameters) == list:
        nargin += 1
    if type(Po) == np.ndarray:
        nargin += 1
    if type(oae) == int:
        nargin += 1
    if type(sc) == dict:
        din=True
    else:
        din=False
        

    if nargin != 5:
        oae = 0
    if not(din):
        sc_pass = 0
    if nargin == 0:
        print("The measurement matrix Y has to be used as argument!!")
    elif nargin == 1:
        n = 2
    elif nargin == 3 or nargin == 4 or nargin == 5:
        if len(parameters) != 9:
            print("The length of parameters vector is not 9 !!")
            print("Default values of hyper-parameters are used instead")
        else:
            initcond, rho, Lambda, epsilon, maxiter, downsampling, parallel, normalization, display = parameters
            if initcond != 1 and initcond != 2 and initcond != 3 and initcond != 4:
                print("The initialization procedure of endmembers matrix is 1,2,3 or 4!")
                print("The default value is considered!")
                initcond = 1
            if rho < 0:
                print("The regularization weight rho cannot be negative")
                print("The default value is considered!")
                rho = 0.1
            if Lambda < 0 or Lambda >= 1:
                print("The entropy weight lambda is limited to [0,1)")
                print("The default value is considered!")
                Lambda = 0
            if epsilon < 0 or epsilon > 0.5:
                print("The threshold epsilon can't be negative or > 0.5")
                print("The default value is considered!")
                epsilon = 1e-3
            if maxiter < 0 and maxiter < 100:
                print("The upper bound maxiter can't be negative or >100")
                print("The default value is considered!")
                maxiter = 20
            if 0 > downsampling > 1:
                print("The downsampling factor cannot be negative or >1")
                print("The default value is considered!")
                downsampling = 0.5
            if parallel != 0 and parallel != 1:
                print("The parallelization parameter is 0 or 1")
                print("The default value is considered!")
                parallel = 0
            if normalization != 0 and normalization != 1:
                print("The normalization parameter is 0 or 1")
                print("The default value is considered!")
                normalization = 1
            if display != 0 and display != 1:
                print("The display parameter is 0 or 1")
                print("The default value is considered")
                display = 0
        if n < 2:
            print("The order of the linear mixture model has to be greater than 2!")
            print("The default value n=2 is considered!")
            n = 2
    if nargin == 4 or nargin == 5:
        if type(Po) != np.ndarray:
            print("The initial end-members Po must be a matrix !!")
            print("The initialization is considered by the maximum cosine difference from mean measurement")
            initcond = 1
        else:
            if Po.shape[0] == Yo.shape[0] and Po.shape[1] == n:
                initcond = 0
            else:
                print("The size of Po must be M x n!!")
                print("The initialization is considered based on the input dataset")
                initcond = 1
    if nargin == 5:
        if oae != 0 and oae != 1:
            print("The assignment of oae is incorrect!!")
            print("The initial end-members Po will be improved iteratively from a selected sample")
            oae = 0
        elif oae == 1 and initcond != 0:
            print("The initial end-members Po is not defined properly!")
            print("Po will be improved iteratively from a selected sample")
            oae = 0
    if din:
        if type(sc) != dict:
            print("The inputs of spatial coherence are incorrect!!")
            print("---- Spatial consistency will not be considered ----")
            sc_pass=0
        elif len(sc) != 5:
            print("The number of inputs is incorrect!!")
            print("---- Spatial consistency will not be considered ----")
            sc_pass=0
        else:
            if sc["mu"]>1:
                print("The value of mu for spatial coherence is unstable.")
                print("The default value of mu=0.01 will be taken.")
                sc["mu"]=0.01
            if sc["nu"]>1:
                print("The value of nu for spatial coherence is unstable.")
                print("The default value of nu=0.0001 will be taken.")
                sc["nu"]=0.0001
            if sc["tau"]>1:
                print("The value of nu for spatial coherence is unstable.")
                print("The default value of tau=0.0001 will be taken.")
                sc["tau"]=0.1
            if sc["dimX"]*sc["dimY"] != Yo.shape[1]:
                print("X and Y dimensions do not correspond to the dimension of the input vector to be analyzed.!!")
                print("---- Spatial consistency will not be considered ----")
                sc_pass=0
    if nargin >= 7:
        print("The number of input arguments is 6 maximum")
        print("Please check the help documentation.")

    # Random downsampling
    if type(Yo) != np.ndarray:
        print("The measurements matrix Y has to be a matrix")
    M, No = Yo.shape
    if M > No:
        print("The number of spatial measurements has to be larger to the number of time samples!")
    N = No#round(No*(1-downsampling))
    # Is = np.random.choice(No, N, replace=False)
    Y = Yo#[:, Is-1]

    # Normalization
    if normalization == 1:
        mYm = np.sum(Y, 0)
        mYmo = np.sum(Yo, 0)
    else:
        mYm = np.ones((1, N), dtype=int)
        mYmo = np.ones((1, No), dtype=int)
    Ym = Y / np.tile(mYm, [M, 1])
    Ymo = Yo / np.tile(mYmo, [M, 1])
    NYm = np.linalg.norm(Ym, 'fro')

    # Selection of Initial Endmembers Matrix
    if initcond == 1 or initcond == 2:
        if initcond == 1:
            Po = np.zeros((M, 1))
            index = 1
            p_max = np.mean(Yo, axis=1)
            Yt = Yo
            Po[:, index-1] = p_max
        elif initcond == 2:
            index = 1
            Y1m = np.sum(abs(Yo), 0)
            y_max = np.max(Y1m)
            Imax = np.argwhere(Y1m == y_max)[0][0]
            y_min = np.min(Y1m)
            I_min = np.argwhere(Y1m == y_min)[0][0]
            p_max = Yo[:, Imax]
            p_min = Yo[:, I_min]
            K = Yo.shape[1]
            II = np.arange(1, K+1)
            condition = np.logical_and(II != II[Imax], II != II[I_min])
            II = np.extract(condition, II)
            Yt = Yo[:, II-1]
            Po = p_max
            index += 1
            Po = np.c_[Po, p_min]
        while index < n:
            y_max = np.zeros((1, index))
            Imax = np.zeros((1, index), dtype=int)
            for j in range(index):
                if j == 0:
                    for i in range(index):
                        e1m = np.around(np.sum(Yt*np.tile(Po[:, i], [Yt.shape[1], 1]).T, 0) /
                                        np.sqrt(np.sum(Yt**2, 0))/np.sqrt(np.sum(Po[:, i]**2, 0)), 4)
                        y_max[j][i] = np.around(np.amin(abs(e1m)), 4)
                        Imax[j][i] = np.where(e1m == y_max[j][i])[0][0]
            ym_max = np.amin(y_max)
            Im_max = np.where(y_max == ym_max)[1][0]
            IImax = Imax[0][Im_max]
            p_max = Yt[:, IImax]
            index += 1
            Po = np.c_[Po, p_max]
            II = np.arange(1, Yt.shape[1]+1)
            II = np.extract(II != IImax+1, II)
            Yt = Yt[:, list(II-1)]
    elif initcond == 3:
        UU, s, VV = np.linalg.svd(Ym.T, full_matrices=False)
        W = VV.T[:, :n]
        Po = W * np.tile(np.sign(W.T@np.ones((M, 1))).T, [M, 1])
    elif initcond == 4:
        Yom = np.mean(Ym, axis=1)
        Yon = Ym-np.tile(Yom, [N, 1]).T
        UU, s, VV = np.linalg.svd(Yon.T, full_matrices=False)
        S = np.diag(s)
        Yo_w = np.linalg.pinv(linalg.sqrtm(S)) @ VV @ Ym
        V, s, u = np.linalg.svd((np.tile(sum(Yo_w * Yo_w), [M, 1]) * Yo_w) @ Yo_w.T, full_matrices=False)
        W = VV.T @ linalg.sqrtm(S)@V[:n, :].T
        Po = W*np.tile(np.sign(W.T@np.ones((M, 1))).T, [M, 1])
    Po = np.where(Po < 0, 0, Po)
    Po = np.where(np.isnan(Po), 0, Po)
    Po = np.where(np.isinf(Po), 0, Po)
    if normalization == 1:
        mPo = np.sum(Po, 0)
        P = Po/np.tile(mPo, [M, 1])
    else:
        P = Po

    # Alternated Least Squares Procedure
    ITER = 1
    J = 1e5
    Jp = 1e6
    a_Time = 0
    p_Time = 0
    tic = time.time()
    if display == 1:
        print("#################################")
        print("EBEAE Linear Unmixing")
        print(f"Model Order = {n}")
        if oae == 1:
            print("Only the abundances are estimated from Po")
        elif oae == 0 and initcond == 0:
            print("The end-members matrix is initialized externally by matrix Po")
        elif oae == 0 and initcond == 1:
            print("Po is constructed based on the maximum cosine difference from mean measurement")
        elif oae == 0 and initcond == 2:
            print("Po is constructed based on the maximum and minimum energy, and largest difference from them")
        elif oae == 0 and initcond == 3:
            print("Po is constructed based on the PCA selection + Rectified Linear Unit")
        elif oae == 0 and initcond == 4:
            print("Po is constructed based on the ICA selection (FOBI) + Rectified Linear Unit")
    Wn=np.zeros([n,Y.shape[1]])
    if sc_pass:
        freeze_support()
        Dh=diags([-1, 1], [0,1],shape=(sc['dimY'],sc['dimY'])).toarray()
        Dh[sc['dimY']-1,:]=0
        Dh=ssp.kron(Dh,np.eye(sc['dimX']),format='csr')
        Dv=diags([-1, 1], [0,1],shape=(sc['dimX'],sc['dimX'])).toarray()
        Dv[sc['dimX']-1,:]=0
        Dv=ssp.kron(np.eye(sc['dimY']),Dv,format='csr')
        tempval= sc['nu']*(Dh.T*Dh+Dv.T*Dv)+ sc['mu']*ssp.eye(Y.shape[1])
        
        # Q, R= linalg.qr(tempval)
        # tqr=time.time()
        # x=np.linalg.solve(tempval,Ay)
        # print(f'Time factorization QR = {time.time()-tqr}')
        
        # del(tempval)
        # Q=ssp.csr_matrix(Q)
        # R=ssp.csr_matrix(R)
        
        while (Jp-J)/Jp >= epsilon and ITER < maxiter and oae == 0 and numerr == 0:
            t_A, outputs_a = abundance_TV(Ym, P, Wn, parallel,sc['mu'])
            Am, numerr = outputs_a
            a_Time += t_A
            Pp = P
            if numerr == 0:
                t_P, outputs_e = endmember(Ym, Am, rho, normalization)
                P, numerr = outputs_e
                p_Time += t_P
            Jp = J
            J = np.linalg.norm(Ym-P@Am, 'fro')
            
            t_w, Wn = dn_abundance(Am,sc,1,tempval, Dh, Dv)
            if J > Jp:
                P = Pp
                break
            if display == 1:
                print(f"Number of iteration = {ITER}")
                print(f"Percentage Estimation Error = {(100*J)/NYm} %")
                print(f"Abundance estimation took {t_A}")
                print(f"Endmember estimation took {t_P}")
            ITER += 1
    
        if numerr == 0:
            t_A, outputs_a = abundance_TV(Ymo, P, Wn, parallel, sc['mu'])
            Am, numerr = outputs_a
            t_w, Wn = dn_abundance(Am,sc,1, tempval, Dh, Dv)
            a_Time += t_A
            toc = time.time()
            elap_time = toc-tic
            if display == 1:
                print(f"Elapsed Time = {elap_time} seconds")
            An = Am
            A = Am * np.tile(mYmo, [n, 1])
            Yh = P @ A
        else:
            print("Please review the problem formulation, not reliable results")
            P = np.array([])
            A = np.array([])
            An = np.array([])
            Yh = np.array([])
        return P, A, An, Wn , Yh#, a_Time, p_Time
        
    else:
        while (Jp-J)/Jp >= epsilon and ITER < maxiter and oae == 0 and numerr == 0:
            t_A, outputs_a = abundance(Ym, P, Lambda, parallel)
            Am, numerr = outputs_a
            a_Time += t_A
            Pp = P
            if numerr == 0:
                t_P, outputs_e = endmember(Ym, Am, rho, normalization)
                P, numerr = outputs_e
                p_Time += t_P
            Jp = J
            J = np.linalg.norm(Ym-P@Am, 'fro')
            if J > Jp:
                P = Pp
                break
            if display == 1:
                print(f"Number of iteration = {ITER}")
                print(f"Percentage Estimation Error = {(100*J)/NYm} %")
                print(f"Abundance estimation took {t_A}")
                print(f"Endmember estimation took {t_P}")
            ITER += 1
    
        if numerr == 0:
            t_A, outputs_a = abundance(Ymo, P, Lambda, parallel)
            Am, numerr = outputs_a
            a_Time += t_A
            toc = time.time()
            elap_time = toc-tic
            if display == 1:
                print(f"Elapsed Time = {elap_time} seconds")
            An = Am
            A = Am * np.tile(mYmo, [n, 1])
            Yh = P @ A
        else:
            print("Please review the problem formulation, not reliable results")
            P = np.array([])
            A = np.array([])
            An = np.array([])
            Yh = np.array([])
        return P, A, An, Wn, Yh#, a_Time, p_Time
