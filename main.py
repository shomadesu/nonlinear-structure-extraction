import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import random

#KSG Estimator
from sklearn.feature_selection import mutual_info_regression
#HSIC
from hyppo.independence import Hsic
#MIC,TIC
from minepy import MINE
#Distance Correlation
import dcor

class structure_extraction():
    def __init__(self,data,measure='ksg',opt_method='powell',repeat_num=10,aug_threshold=1e-2):
        self.data = data
        self.measure = measure
        self.opt_method = opt_method
        self.repeat_num = repeat_num
        self.aug_threshold = aug_threshold

    def caluculate_dependence(self,x,y):
        if self.measure == 'pearson':
            return -abs(np.corrcoef(x,y)[0,1])
        if self.measure == 'kendall':
            tau, p_value = sp.stats.kendalltau(x, y)
            return  -abs(tau)
        if self.measure == 'spearman':
            correlation, pvalue = sp.stats.spearmanr(x,y)
            return  -abs(correlation)
        if self.measure == 'dcor':
            return -dcor.distance_correlation(x,y)
        if self.measure == 'ksg':
            x = x.reshape(-1,1)
            mi = mutual_info_regression(x, y)
            return -(1-2**(-2*mi))[0]
        if self.measure == 'mic':
            mine = MINE()
            mine.compute_score(x,y)
            return -mine.mic()
        if self.measure == 'tic':
            mine = MINE()
            mine.compute_score(x,y)
            return -mine.tic(norm=True)
        if self.measure == 'mic_e':
            mine = MINE(est="mic_e")
            mine.compute_score(x,y)
            return -mine.mic()
        if self.measure == 'tic_e':
            mine = MINE(est="mic_e")
            mine.compute_score(x,y)
            return -mine.tic(norm=True)
        if self.measure == 'hsic':
            stat, _ = Hsic().test(x, y)
            return -stat

    def caluculate_projection(self,alpha,beta):
        x = np.dot(self.data,alpha)
        y = np.dot(self.data,beta)
        return x,y

    def optimize(self):
        alpha_beta_best = [[0],[0]]
        dependence_best = 0
        for i in range(self.repeat_num):
            alpha_beta = np.array([random.uniform(-1,1) for i in range(0,len(self.data.T)*2)])
            p = 1
            u = [0,0,0]
            while True:
                if self.constraint(alpha_beta,p) < self.aug_threshold:
                    break
                alpha_beta = sp.optimize.minimize(self.objective_func,x0=alpha_beta,method=self.opt_method,args=(u,p)).x
                u = self.update_u(alpha_beta,u,p)
                p = p*2
            alpha = alpha_beta[0:len(self.data.T)]
            beta = alpha_beta[len(self.data.T):len(self.data.T)*2]
            x,y = self.caluculate_projection(alpha,beta)
            dependence = self.caluculate_dependence(x,y)
            if dependence < dependence_best:
                alpha_beta_best = [alpha,beta]
                dependence_best = dependence
        return alpha_beta_best, dependence_best

    def objective_func(self,alpha_beta,u,p):
        alpha = alpha_beta[0:len(self.data.T)]
        beta = alpha_beta[len(self.data.T):len(self.data.T)*2]
        x,y = self.caluculate_projection(alpha,beta)
        return self.caluculate_dependence(x,y) + u[0]*(np.dot(alpha,beta)) + u[1]*(np.linalg.norm(alpha)-1) +u[2]*(np.linalg.norm(beta)-1) + (p/2)*((np.dot(alpha,beta))**2 + (np.linalg.norm(alpha)-1)**2 + (np.linalg.norm(beta)-1)**2)

    def constraint(self,alpha_beta,p):
        alpha = alpha_beta[0:len(self.data.T)]
        beta = alpha_beta[len(self.data.T):len(self.data.T)*2]
        return (p/2)*((np.dot(alpha,beta))**2 + (np.linalg.norm(alpha)-1)**2 + (np.linalg.norm(beta)-1)**2)

    def update_u(self,alpha_beta,u,p):
        alpha = alpha_beta[0:len(self.data.T)]
        beta = alpha_beta[len(self.data.T):len(self.data.T)*2]
        tmp = [p*(np.dot(alpha,beta)), p*(np.linalg.norm(alpha)-1), p*(np.linalg.norm(beta)-1)]
        return  [[xx + yy for xx,yy in zip(x,y)] for x,y in zip([u],[tmp])][0]

    def fig_plot(self):
        alpha_beta, dependence = self.optimize()
        alpha = alpha_beta[0]
        beta = alpha_beta[1]
        proj = pd.DataFrame(self.caluculate_projection(alpha,beta)).T
        proj.columns = ['x','y']

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(proj.x, proj.y, color="#007FB1",lw=3)
        ax.set_xlabel("px",labelpad=10, fontsize=24)
        ax.set_ylabel("py",labelpad=10, fontsize=24)
        ax.xaxis.set_label_coords(0.5, 0)
        ax.yaxis.set_label_coords(0, 0.5)
        plt.show()

        return alpha, beta, dependence, proj, fig
