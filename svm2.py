import numpy as np
from tqdm import tqdm
import sys

# 增大递归深度

class SVM:
    def __init__(self, X, XL, T, TL, C) -> None:
        super().__init__()
        self.X = X
        self.XL = XL
        self.T = T
        self.TL = TL
        self.K = np.zeros((X.shape[0], X.shape[0]))
        self.KT = np.zeros((T.shape[0], X.shape[0]))
        self.alpha = np.zeros(X.shape[0])
        self.E = np.zeros(X.shape[0])
        self.b = 0
        self.C = C

    def kernel_process(self, p):
        """用核方法处理数据并存储到K中"""

        if p == 0:  # 线性核
            self.K = np.dot(self.X, self.X.T)

        if p == 1:  # 多项式核
            d = 1
            self.K = np.dot(self.X, self.X.T) ** d

        if p == 2:  # 高斯核
            sigma = 1
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[0]):
                    self.K[i][j] = np.exp(-(np.linalg.norm(self.X[i] - self.X[j], ord=2) ** 2) / (2 * sigma))

        if p == 3:  # 拉普拉斯核
            sigma2 = 1
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[0]):
                    self.K[i][j] = np.exp(-np.linalg.norm(self.X[i] - self.X[j], ord=2) / (2 * sigma2))

        if p == 4:  # Sigmoid核
            beta = 1
            theta = -1
            self.K = np.tanh(beta * np.dot(self.X, self.X.T) + theta)

    def update_E(self):
        """更新E序列"""
        for i in range(self.X.shape[0]):
            self.E[i] = np.dot(np.multiply(self.alpha, self.XL),
                               self.K[:, i:i + 1].reshape(self.X.shape[0]).T) + self.b - self.XL[i]

    def one_step_smo(self, i, j, H, L):
        """执行smo算法的核心部分"""

        # 更新一下E序列
        self.update_E()

        # 求解新的alpha_i和j
        eta = self.K[i][i] + self.K[j][j] - 2 * self.K[i][j]
        alpha_j = self.alpha[j] + self.XL[j] * (self.E[i] - self.E[j]) / eta
        alpha_j = min(H, alpha_j)
        alpha_j = max(L, alpha_j)

        alpha_i = self.alpha[i] + self.XL[i] * self.XL[j] * (self.alpha[j] - alpha_j)

        # 更新b值
        bi = -self.E[i] - self.XL[i] * self.K[i][i] * (alpha_i - self.alpha[i]) - self.XL[j] * self.K[j][i] * (
                alpha_j - self.alpha[j]) + self.b
        bj = -self.E[j] - self.XL[i] * self.K[i][j] * (alpha_i - self.alpha[i]) - self.XL[j] * self.K[j][j] * (
                alpha_j - self.alpha[j]) + self.b

        # b的更新很讲究，第一次就是错在这了
        if 0 < alpha_i < self.C:
            self.b = bi
        elif 0 < alpha_j < self.C:
            self.b = bj
        else:
            self.b = (bi + bj) / 2

        self.alpha[i] = alpha_i
        self.alpha[j] = alpha_j

    def gen_i_j(self):
        """选择两个变量进行更新"""
        vis_i = np.zeros(self.X.shape[0])
        vis_j = np.zeros(self.X.shape[0])
        # 如果i全满了，则随机选

        eps = 0.01
        max_err = 0

        # 按照违背KKT条件的程度遴选第一个变量
        for k in range(self.X.shape[0]):
            fxi = self.E[k] + self.XL[k]
            if (0 < self.alpha[k] < self.C) and np.abs(self.XL[k] * fxi - 1) > eps:
                err = np.abs(self.XL[k] * fxi - 1) - eps
                if err > max_err and vis_i[k] == 0:
                    max_err = err
                    i = k
            elif np.abs(self.alpha[k] - self.C) < eps and self.XL[k] * fxi > 1 + eps:
                err = self.XL[k] * fxi - 1 - eps
                if err > max_err and vis_i[k] == 0:
                    max_err = err
                    i = k
            elif np.abs(self.alpha[k]) < eps and self.XL[k] * fxi < 1 - eps:
                err = 1 - eps - self.XL[k] * fxi
                if err > max_err and vis_i[k] == 0:
                    max_err = err
                    i = k

        dif = 0
        for k in range(self.X.shape[0]):
            tmp = abs(self.E[k]-self.E[i])
            if tmp > dif and vis_j[k] == 0:
                dif = tmp
                j = k

        # if self.E[i] > 0:
        #     j = np.argmin(self.E)
        # else:
        #     j = np.argmax(self.E)

        if self.XL[i] != self.XL[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[j] + self.alpha[i] - self.C)
            H = min(self.C, self.alpha[j] + self.alpha[i])

        # # 求最优值的约束条件
        # vis = np.zeros(self.X.shape[0])
        # vis[i] = 1
        #
        # while True:
        #     if self.XL[i] != self.XL[j]:
        #         L = max(0, self.alpha[j] - self.alpha[i])
        #         H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        #     else:
        #         L = max(0, self.alpha[j] + self.alpha[i] - self.C)
        #         H = min(self.C, self.alpha[j] + self.alpha[i])
        #
        #     # H==L的情况需要特殊处理
        #     if H == L:
        #         # 如果所有的j都不能使H！=L，则随机初始化i和j
        #         if (vis == 1).all():
        #             i = np.random.randint(0, self.X.shape[0])
        #             j = i
        #             # 给j随机赋一个和i不同的随机值
        #             while j == i:
        #                 j = np.random.randint(0, self.X.shape[0])
        #             # 清空vis数组
        #             vis = np.zeros(self.X.shape[0])
        #             vis[i] = 1
        #         else:
        #             vis[j] = 1
        #             while vis[j] == 1:
        #                 j = np.random.randint(0, self.X.shape[0])
        #     else:
        #         break

        return i, j, H, L

    def cal_err(self):
        """计算总误差值"""
        eps = 0.01
        err = 0
        for k in range(self.X.shape[0]):
            fxi = self.E[k] + self.XL[k]
            if (0 < self.alpha[k] < self.C) and np.abs(self.XL[k] * fxi - 1) > eps:
                err += np.abs(self.XL[k] * fxi - 1) - eps
            elif np.abs(self.alpha[k] - self.C) < eps and self.XL[k] * fxi > 1 + eps:
                err += self.XL[k] * fxi - 1 - eps
            elif np.abs(self.alpha[k]) < eps and self.XL[k] * fxi < 1 - eps:
                err += 1 - eps - self.XL[k] * fxi

        return err

    # def stop_machine(self, err_b):
    #     """停机条件"""
    #     eps = 0.01
    #     for k in range(self.X.shape[0]):
    #         fxi = self.E[k] + self.XL[k]
    #         if (0 < self.alpha[k] < self.C) and np.abs(self.XL[k] * fxi - 1) > eps:
    #             return True
    #         elif np.abs(self.alpha[k] - self.C) < eps and self.XL[k] * fxi > 1 + eps:
    #             return True
    #         elif np.abs(self.alpha[k]) < eps and self.XL[k] * fxi < 1 - eps:
    #             return True
    #     return False

    def train_svm(self, p):
        """训练svm"""
        err_b = float('inf')
        eps = 1E-5
        self.kernel_process(p)
        self.update_E()
        for k in tqdm(range(20000)):
            i, j, H, L = self.gen_i_j()
            self.one_step_smo(i, j, H, L)
            if k != 0 and k % 100 == 0:
                self.process_of_test_data(p)
                if self.test_model(p) > 0.5:
                    break

            # err_n = self.cal_err()
            # if np.abs(err_n - err_b) < eps:
            #     break
            # else:
            #     err_b = err_n

    def process_of_test_data(self, p):
        """处理测试数据"""
        if p == 0:
            self.KT = np.dot(self.T, self.X.T)

        if p == 1:
            d = 1
            self.KT = np.dot(self.T, self.X.T) ** d

        if p == 2:
            sigma = 1
            for i in range(self.T.shape[0]):
                for j in range(self.X.shape[0]):
                    self.KT[i][j] = np.exp(-(np.linalg.norm(self.T[i] - self.X[j], ord=2) ** 2) / (2 * sigma))

        if p == 3:
            sigma2 = 1
            for i in range(self.T.shape[0]):
                for j in range(self.X.shape[0]):
                    self.KT[i][j] = np.exp(-np.linalg.norm(self.T[i] - self.X[j], ord=2) / (2 * sigma2))

        if p == 4:
            beta = 1
            theta = -1
            self.KT = np.tanh(beta * np.dot(self.T, self.X.T) + theta)

    def get_result(self, p):
        """得出结果"""
        # 注意：要先训练模型并处理测试数据后再调用此方法
        res = np.zeros((self.T.shape[0]))
        for i in range(self.T.shape[0]):
            res[i] = np.dot(np.multiply(self.alpha, self.XL), self.KT[i:i + 1, :].reshape(self.X.shape[0]).T) + self.b

        return res

    def test_model(self, p):
        """用于模型的检验"""
        # 注意：要先训练模型并处理测试数据后再调用此方法
        cnt = 0
        for i in range(self.T.shape[0]):
            if (np.dot(np.multiply(self.alpha, self.XL), self.KT[i:i + 1, :].reshape(self.X.shape[0]).T) + self.b) * \
                    self.TL[i] > 0:
                cnt += 1

        return float(cnt / self.T.shape[0])

        # # 计算模型的w用于可视化
        # w = np.dot(np.multiply(self.alpha, self.XL), self.X)
        # return w, self.b
