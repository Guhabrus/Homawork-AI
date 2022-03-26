import numpy as np
import matplotlib.pyplot as plt
import math

class EM:
    def __init__ (self,X:np.ndarray,N= 10000, k=2):
        self._N = N
        self._k = k
        self._X = X
        self._m = X.shape[1]
        self._n = X.shape[0]
        self._sigma = np.array(([1., 0.],[0., 1.], [1., 0.],[0., 1.]))
        self._sigma = self._sigma.reshape(k, self._m, self._m)
        self._w = np.array([float(1./k), float(1./k)])
        self._mu  = np.array((np.mean(self._X[np.random.choice(self._n, int(self._n/k))], axis = 0), np.mean(self._X[np.random.choice(self._n, int(self._n/k))], axis = 0)))

        # self._mu  = np.array([1,1])#np.array((np.mean(X[np.random.choice(self._n, self._n/k)], axis = 0), np.mean(X[np.random.choice(self._n, self._n/k)], axis = 0)))

    def e_step(self):
     
        pj_xi = []
        for j in range(self._k):
            det_sigma_j = np.linalg.det(self._sigma [j])
            factor_1 = 1 / (((2 * math.pi)**(self._k/2)) * ((det_sigma_j)**0.5))
            factor_2 = []
            for i in  self._X:
                factor_2.append(math.e**float(-0.5 * np.matrix(i - self._mu [j]) * np.matrix(np.linalg.inv(self._sigma [j])) * np.matrix(i - self._mu [j]).T))
            pj_xi.append(factor_1 * np.array(factor_2))
        pj_xi = np.array(pj_xi)
        
        
        pj_xi_w = []
        for j in range(self._k):
            pj_xi_w.append(pj_xi[j] * self._w[j])
        pj_xi_w = np.array(pj_xi_w)
        # print("dcd = ",pj_xi_w)
        
        sum_pj_xi_w = np.sum(pj_xi_w, axis = 0)
        
        
        proba_xi = []
        for j in range(self._k):
            proba_xi.append(pj_xi_w[j]/ sum_pj_xi_w)
            # print(f"1 = {pj_xi_w[j]} 2 = {sum_pj_xi_w} 3 = {pj_xi_w[j]/ sum_pj_xi_w}")
        
        return np.array(proba_xi)

    def x_new(self, proba_xi):
        X1_new_ind = []
        X2_new_ind = []
        X_answers = []

        count = 0
        for x in proba_xi[0]:
            if x >= 0.5:
                X1_new_ind.append(count)
                X_answers.append(1)
            else:
                X2_new_ind.append(count)
                X_answers.append(2)
            count += 1
        
        return X1_new_ind, X2_new_ind, X_answers


    def m_step(self, proba_xi):
        w_new = np.sum(proba_xi, axis = 1) / self._N 
        
        print(w_new)
        
        mu_new = (np.array((np.matrix(proba_xi) * np.matrix(self._X))).T / np.sum(proba_xi, axis = 1)).T
        
        # рассчитаем дисперсии
        cov_new = []
        for mu in range(mu_new.shape[0]):
            X_cd = []
            X_cd_proba = []
            count = 0
            for x_i in  self._X:
                cd = np.array(x_i - mu_new[mu])
                X_cd.append(cd)
                X_cd_proba.append(cd * proba_xi[mu][count])
                count += 1
            X_cd = np.array(X_cd)
            X_cd = X_cd.reshape(self._N , self._m)
            X_cd_proba = np.array(X_cd_proba)
            X_cd_proba = X_cd_proba.reshape(self._N , self._m)

            cov_new.append(np.matrix(X_cd.T) * np.matrix(X_cd_proba))
        cov_new = np.array((np.array(cov_new) / (np.sum(proba_xi, axis = 1)-1)))
       
        if cov_new[0][0][1] < 0:
            cov_new[0][0][1] = 0
        if cov_new[0][1][0] < 0:
            cov_new[0][1][0] = 0
        
        if cov_new[1][0][1] < 0:
            cov_new[1][0][1] = 0
        if cov_new[1][1][0] < 0:
            cov_new[1][1][0] = 0
        
        # рассчитаем стандартное отклонение
        sigma_new = cov_new**0.5
        return w_new, mu_new, sigma_new


    def fit(self, c_iteration):
        for i in range(c_iteration):
            proba_xi = self.e_step()
       
            self._w, self._mu, self._sigma = self.m_step(proba_xi)
            X1_new_ind, X2_new_ind, X_answers = self.x_new(proba_xi)
            print('Итерация №', i+1)
            
            print('Матрица значений математических ожиданий')
            print(self._mu)

            print('Матрица значений стандартных отклонений')
            print(self._sigma)
            
            # print('Доля правильно распознанных изделий')
            # print(round(accuracy_score(y, X_answers),3))
            
        plt.figure(figsize=(16, 6))  
        plt.plot(
            self._X[X1_new_ind,0], X[X1_new_ind,1], 'o', alpha = 0.7, color='sandybrown', label = 'Produced on machine #1')
        plt.plot(
            self._X[X2_new_ind,0], X[X2_new_ind,1], 'o', alpha = 0.45, color = 'darkblue', label = 'Produced on machine #2')
        plt.plot(self._mu[0][0], self._mu[0][1], 'o', markersize = 16, color = 'r', label = 'Mu 1')
        plt.plot(self._mu[1][0], self._mu[1][1], 'o',  markersize = 16, color = 'b', label = 'Mu 2')
        plt.xlabel('Diameter')
        plt.ylabel('Weight')
        plt.legend()
        plt.show()


k = 2       #количество признаков
N1 = 6000
N2 = 4000
N = N1+N2
m = 2       #количеств опризнаков

diametr1 = 64
mass1    = 14
sigma_d1  = 3.5
sigma_m1 = 1

diametr2 = 52
mass2    = 9.5
sigma_d2  = 2
sigma_m2 = 0.7
X = np.zeros((N1+N2, m))

X[:N1, 0] = np.random.normal(diametr1, sigma_d1, N1)
X[:N1, 1] = np.random.normal(mass1, sigma_m1, N1)

X[N1:N, 0] = np.random.normal(diametr2, sigma_d2, N2)
X[N1:N, 1] = np.random.normal(mass2, sigma_m2, N2)


Y = np.zeros((N))
Y[:N1] = 1
Y[N1:N] =2

model = EM(X, 10000)
model.fit(15)