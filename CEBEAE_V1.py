# Extended Blind Endmembers and Abundances Extraction (EBEAE) Algorithm
import time
import numpy as np
from scipy import linalg


def performance(fn):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        # print(f'Function {fn.__name__} took {t2-t1} s')
        return t2 - t1, result
    return wrapper


class EBEAE:
    def __init__(self, Yo=[], n=2, initcond = 1, epsilon = 1e-3, maxiter = 20, downsampling = 0, parallel = 0, normalization = 1, display = 0, Po=[], oae=0):
        """
        Estimation of Optimal Endmembers and Abundances in Linear Mixture Model
        Input Arguments:
            Y = matrix of measurements (MxN)
            n = order of linear mixture model
            initcond = initialization of endmembers matrix {1,2,3,4}
                                      (1) Maximum cosine difference from mean measurement (default)
                                      (2) Maximum and minimum energy, and largest distance from them
                                      (3) PCA selection + Rectified Linear Unit
                                      (4) ICA selection (FOBI) + Rectified Linear Unit
            epsilon = threshold for convergence in ALS method (default epsilon=1e-3)
            maxiter = maximum number of iterations in ALS method (default maxiter=20)
            downsampling = percentage of random downsampling in endmember estimation [0,1) (default downsampling=0)
            parallel = implement parallel computation of abundances (0->NO or 1->YES) (default parallel=0)
            normalization = normalization of estimated end-members (0->NO or 1->YES) (default normalization=1)
            display = show progress of iterative optimization process (0->NO or 1->YES) (default display=0)
            Po = initial end-member matrix (Mxn)
            oae = only optimal abundance estimation with Po (0 -> NO or 1 -> YES) (default oae = 0)
        """
        self.numerr = 0
        self.rho = 0.1
        self.Lambda = 0
        self.oae=0
        
        if not len(Yo):
            print("The measurement matrix Y has to be used as argument!!")
            return 0
        else:
            if type(Yo) != np.ndarray:
                print("The measurements matrix Y has to be a matrix")
            else:
                self.Yo=Yo
        if n<2:
            print('The order of the linear mixture model has to greater than 2!')
            print('The default value n=2 is considered!')
            self.n=2
        else:
            self.n=n
        if initcond != 1 and initcond != 2 and initcond != 3 and initcond != 4:
            print("The initialization procedure of endmembers matrix is 1,2,3 or 4!")
            print("The default value is considered!")
            self.initcond = 1
        else:
            self.initcond = initcond
        if epsilon < 0 or epsilon > 0.5:
            print("The threshold epsilon can't be negative or > 0.5")
            print("The default value is considered!")
            self.epsilon = 1e-3
        else:
            self.epsilon = epsilon
        if maxiter < 0 and maxiter < 100:
            print("The upper bound maxiter can't be negative or >100")
            print("The default value is considered!")
            self.maxiter = 20
        else:
            self.maxiter = maxiter
        if 0 > downsampling > 1:
            print("The downsampling factor cannot be negative or >1")
            print("The default value is considered!")
            self.downsampling = 0
        else:
            self.downsampling =downsampling
        if parallel != 0 and parallel != 1:
            print("The parallelization parameter is 0 or 1")
            print("The default value is considered!")
            self.parallel = 0
        else:
            self.parallel =parallel
        if normalization != 0 and normalization != 1:
            print("The normalization parameter is 0 or 1")
            print("The default value is considered!")
            self.normalization = 1
        else:
            self.normalization = normalization
        if display != 0 and display != 1:
            print("The display parameter is 0 or 1")
            print("The default value is considered")
            self.display = 0
        else:
            self.display = display

        if len(Po):
            if type(Po) != np.ndarray:
                print("The initial end-members Po must be a matrix !!")
                print("The initialization is considered by the maximum cosine difference from mean measurement")
                self.initcond = 1
            else:
                if Po.shape[0] == Yo.shape[0] and Po.shape[1] == n:
                    self.initcond = 0
                    self.Po = Po
                else:
                    print("The size of Po must be M x n!!")
                    print("The initialization is considered based on the input dataset")
                    self.initcond = 1
        if oae != 0 and oae != 1:
            print("The assignment of oae is incorrect!!")
            print("The initial end-members Po will be improved iteratively from a selected sample")
            self.oae = 0
        elif oae == 1 and self.initcond != 0:
            print("The initial end-members Po is not defined properly!")
            print("Po will be improved iteratively from a selected sample")
            self.oae = 0
        elif oae == 1 and self.initcond == 0:
            self.oae = 1
        
        
        
        M, No = Yo.shape
        if M > No:
            print("The number of spatial measurements has to be larger to the number of time samples!")
        N = round(No*(1-downsampling))
        Is = np.random.choice(No, N, replace=False)
        Y = Yo[:, Is-1]
        
    
        # Normalization
        if self.normalization == 1:
            mYm = np.sum(Y, 0)
            self.mYmo = np.sum(Yo, 0)
        else:
            mYm = np.ones((1, N), dtype=int)
            self.mYmo = np.ones((1, No), dtype=int)
        self.Ym = Y / np.tile(mYm, [M, 1])
        self.Ymo = Yo / np.tile(self.mYmo, [M, 1])
        self.NYm = np.linalg.norm(self.Ym, 'fro')
        self.M, self.N = self.Ym.shape
        
        self.initializationEM()


    def initializationEM(self):
        
        # Selection of Initial Endmembers Matrix
        if self.initcond == 1 or self.initcond == 2:
            if self.initcond == 1:
                self.Po = np.zeros((self.M, 1))
                index = 1
                p_max = np.mean(self.Yo, axis=1)
                Yt = self.Yo
                self.Po[:, index-1] = p_max
            elif self.initcond == 2:
                index = 1
                Y1m = np.sum(abs(self.Yo), 0)
                y_max = np.max(Y1m)
                Imax = np.argwhere(Y1m == y_max)[0][0]
                y_min = np.min(Y1m)
                I_min = np.argwhere(Y1m == y_min)[0][0]
                p_max = self.Yo[:, Imax]
                p_min = self.Yo[:, I_min]
                II = np.arange(1, self.N)
                condition = np.logical_and(II != II[Imax], II != II[I_min])
                II = np.extract(condition, II)
                Yt = self.Yo[:, II-1]
                self.Po = p_max
                index += 1
                self.Po = np.c_[self.Po, p_min]
            while index < self.n:
                y_max = np.zeros((1, index))
                Imax = np.zeros((1, index), dtype=int)
                for j in range(index):
                    if j == 0:
                        for i in range(index):
                            e1m = np.around(np.sum(Yt*np.tile(self.Po[:, i], [Yt.shape[1], 1]).T, 0) /
                                            np.sqrt(np.sum(Yt**2, 0))/np.sqrt(np.sum(self.Po[:, i]**2, 0)), 4)
                            y_max[j][i] = np.around(np.amin(abs(e1m)), 4)
                            Imax[j][i] = np.where(e1m == y_max[j][i])[0][0]
                ym_max = np.amin(y_max)
                Im_max = np.where(y_max == ym_max)[1][0]
                IImax = Imax[0][Im_max]
                p_max = Yt[:, IImax]
                index += 1
                self.Po = np.c_[self.Po, p_max]
                II = np.arange(1, Yt.shape[1]+1)
                II = np.extract(II != IImax+1, II)
                Yt = Yt[:, list(II-1)]
        elif self.initcond == 3:
            UU, s, VV = np.linalg.svd(self.Ym.T, full_matrices=False)
            W = VV.T[:, :self.n]
            self.Po = W * np.tile(np.sign(W.T@np.ones((self.M, 1))).T, [self.M, 1])
        elif self.initcond == 4:
            Yom = np.mean(self.Ym, axis=1)
            Yon = self.Ym-np.tile(Yom, [self.N, 1]).T
            UU, s, VV = np.linalg.svd(Yon.T, full_matrices=False)
            S = np.diag(s)
            Yo_w = np.linalg.pinv(linalg.sqrtm(S)) @ VV @ self.Ym
            V, s, u = np.linalg.svd((np.tile(sum(Yo_w * Yo_w), [self.M, 1]) * Yo_w) @ Yo_w.T, full_matrices=False)
            W = VV.T @ linalg.sqrtm(S)@V[:self.n, :].T
            self.Po = W*np.tile(np.sign(W.T@np.ones((self.M, 1))).T, [self.M, 1])
        self.Po = np.where(self.Po < 0, 0, self.Po)
        self.Po = np.where(np.isnan(self.Po), 0, self.Po)
        self.Po = np.where(np.isinf(self.Po), 0, self.Po)
        if self.normalization == 1:
            mPo = np.sum(self.Po, 0)
            self.P = self.Po/np.tile(mPo, [self.M, 1])
        else:
            self.P = self.Po

    @performance
    def abundance(self, Y):
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
        n = self.P.shape[1]
        self.A = np.zeros((n, self.N))
        if self.P.shape[0] != self.M:
            print("ERROR: the number of rows in Y and P does not match")
            self.numerr = 1
    
        # Compute fixed vectors and matrices
        c = np.ones((n, 1))
        d = 1  # Lagrange Multiplier for equality restriction
        Go = self.P.T @ self.P
        w, v = np.linalg.eig(Go)
        l_min = np.amin(w)
        G = Go-np.eye(n)*l_min*self.Lambda
        Lambda=self.Lambda
        while (1/np.linalg.cond(G, 1)) < 1e-6:
            Lambda = Lambda/2
            G = Go-np.eye(n)*l_min*Lambda
            if Lambda < 1e-6:
                print("Unstable numerical results in abundances estimation, update rho!!")
                self.numerr = 1
        Gi = np.linalg.pinv(G)
        T1 = Gi@c
        T2 = c.T@T1
    
        # Start Computation of Abundances
        for k in range(self.N):
            yk = np.c_[Y[:, k]]
            byk = float(yk.T@yk)
            bk = self.P.T@yk
    
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
            self.A[:, k] = np.c_[ak].T


    @performance
    def endmember(self, Y):
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
        M, K = Y.shape
        R = sum(self.n-np.array(range(1, self.n)))
        W = np.tile((1/K/sum(Y**2)), [self.n, 1]).T
        if Y.shape[1] != self.A.shape[1]:
            print("ERROR: the number of columns in Y and A does not match")
            self.numerr = 1
        o = (self.n * np.eye(self.n)-np.ones((self.n, self.n)))
        n1 = (np.ones((self.n, 1)))
        m1 = (np.ones((self.M, 1)))
    
        # Construct Optimal Endmembers Matrix
        T0 = (self.A @ (W*self.A.T)+self.rho*np.divide(o, R))
        rho=self.rho
        while 1/np.linalg.cond(T0, 1) < 1e-6:
            rho = rho/10
            T0 = (self.A @ (W*self.A.T)+rho*np.divide(o, R))
            if rho < 1e-6:
                print("Unstable numerical results in endmembers estimation, update rho!!")
                self.numerr = 1
        V = (np.eye(self.n) @ np.linalg.pinv(T0))
        T2 = (Y @ (W*self.A.T) @ V)
        if self.normalization == 1:
            T1 = (np.eye(self.M)-(1/self.M)*(m1 @ m1.T))
            T3 = ((1/self.M)*m1 @ n1.T)
            P_est = T1 @ T2 + T3
        else:
            P_est = T2
    
        # Evaluate and Project Negative Elements
        P_est = np.where(P_est < 0, 0, P_est)
        P_est = np.where(np.isnan(P_est), 0, P_est)
        P_est = np.where(np.isinf(P_est), 0, P_est)
    
        # Normalize Optimal Solution
        if self.normalization == 1:
            P_sum = np.sum(P_est, 0)
            self.P = P_est/np.tile(P_sum, [self.M, 1])
        else:
            self.P = P_est


    @performance
    def evaluate(self, rho=0.1, Lambda=0):
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
        
        if rho < 0:
            print("The regularization weight rho cannot be negative")
            print("The default value is considered!")
        else:
            self.rho = rho
        if Lambda < 0 or Lambda >= 1:
            print("The entropy weight lambda is limited to [0,1)")
            print("The default value is considered!")
        else:
            self.Lambda = Lambda
            
    
        # Alternated Least Squares Procedure
        ITER = 1
        J = 1e5
        Jp = 1e6
        a_Time = 0
        p_Time = 0
        tic = time.time()
        if self.display == 1:
            print("#################################")
            print("EBEAE Linear Unmixing")
            print(f"Model Order = {self.n}")
            if self.oae == 1:
                print("Only the abundances are estimated from Po")
            elif self.oae == 0 and self.initcond == 0:
                print("The end-members matrix is initialized externally by matrix Po")
            elif self.oae == 0 and self.initcond == 1:
                print("Po is constructed based on the maximum cosine difference from mean measurement")
            elif self.oae == 0 and self.initcond == 2:
                print("Po is constructed based on the maximum and minimum energy, and largest difference from them")
            elif self.oae == 0 and self.initcond == 3:
                print("Po is constructed based on the PCA selection + Rectified Linear Unit")
            elif self.oae == 0 and self.initcond == 4:
                print("Po is constructed based on the ICA selection (FOBI) + Rectified Linear Unit")
    
        while (Jp-J)/Jp >= self.epsilon and ITER < self.maxiter and self.oae == 0 and self.numerr == 0:
            t_A = self.abundance(self.Ym)
            a_Time += t_A[0]
            Pp=self.P
            if self.numerr == 0:
                t_P = self.endmember(self.Ym)
                p_Time += t_P[0]
            Jp = J
            J = np.linalg.norm(self.Ym-self.P@self.A, 'fro')
            if J > Jp:
                self.P = Pp
                break
            if self.display == 1:
                print(f"Number of iteration = {ITER}")
                print(f"Percentage Estimation Error = {(100*J)/self.NYm} %")
                print(f"Abundance estimation took {t_A}")
                print(f"Endmember estimation took {t_P}")
            ITER += 1
    
        if self.numerr == 0:
            t_A = self.abundance(self.Ymo)
            a_Time += t_A[0]
            toc = time.time()
            elap_time = toc-tic
            if self.display == 1:
                print(f"Elapsed Time = {elap_time} seconds")
            self.An = self.A
            self.A = self.A * np.tile(self.mYmo, [self.n, 1])
            self.Yh = self.P @ self.A
        else:
            print("Please review the problem formulation, not reliable results")
            self.P = np.array([])
            self.A = np.array([])
            self.An = np.array([])
            self.Yh = np.array([])
        return self.P, self.A, self.An, self.Yh, a_Time, p_Time
