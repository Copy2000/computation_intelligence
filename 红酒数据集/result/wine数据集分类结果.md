# wine数据集分类结果

## PSO

### 算法大致步骤：

* 初始化
* 更新

### 代码实现设置的条件：

```python
MAX_Generation = 10		#迭代次数
Population = 10			#种群数量
dimension = 3
v_low = -1
v_high = 1
pso = PSO(dimension, MAX_Generation, Population,
              BOUNDS_LOW, BOUNDS_HIGH, v_low, v_high) #实现的class
```

### 更新步骤的核心代码:

网上搜一下，其中w自身权重系数（记不清了），c1是个体学习系数，c2是群落学习系数

```python
c1 = 2.0  # 学习因子
c2 = 2.0
w = 0.8
# 更新速度(核心公式)
self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
# 速度限制
for j in range(self.dimension):
    if self.v[i][j] < self.v_low:
        self.v[i][j] = self.v_low
    if self.v[i][j] > self.v_high:
        self.v[i][j] = self.v_high

# 更新位置
self.x[i] = self.x[i] + self.v[i]
# 位置限制
for j in range(self.dimension):
    if self.x[i][j] < self.bound[0][j]:
        self.x[i][j] = self.bound[0][j]
    if self.x[i][j] > self.bound[1][j]:
        self.x[i][j] = self.bound[1][j]
```

大致的意思就是通过公式更新速度和位置，同时对更新后的速度与位置进行修正，因为更新的位置一定是在一定的范围之内的。

### 结果：

1. > MAX_Generation = 10		#迭代次数
   > Population = 10			#种群数量
   > c1 = 2.0  # 学习因子
   > c2 = 2.0
   > w = 0.8
   >
   > ---------------------------
   
   > n_estimators'=97.59544738, 'learning_rate'=0.93788895, 'algorithm'=SAMME
   > 0.9830158730158731
   > 当前的最佳适应度：0.9830158730158731
   > time cost:       175.3606903553009
   >
   > ![image-20220523224042529](wine数据集分类结果/image-20220523224042529.png)
   
2. > MAX_Generation = 10		#迭代次数
   > Population = 10			#种群数量
   > c1 = 2.0  # 学习因子
   > c2 = 2.0
   > w = 0.8

   > 当前最佳位置：[22.63664273  0.7580015   0.39425009]
   > 0.9831746031746033
   > 当前的最佳适应度：0.9831746031746033
   > time cost:        1097.9579124450684      s
   >
   > ![image-20220523235957152](wine数据集分类结果/image-20220523235957152.png)

​	