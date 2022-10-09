# Nonlinear Structure Extraction

## Algorithm Description
This method transforms multivariate data into 2-dimensional data, and extracts linear and nonlinear structure. A full description of the algorithm is given in the [2022 paper](https://link.springer.com/article/10.1007/s42081-022-00177-9) by Ishimoto, S., Minami, H., & Mizuta, M.

![](https://user-images.githubusercontent.com/46952903/194743710-fb9542b3-d9e9-4d07-b7ec-57ff500d7639.png)

## Usage

~~~

#define target data
#quadratic structure with noise
x= np.linspace(-1/2,1/2,100)
y = x**2
z = np.array([random.uniform(0, 1) for i in range(0,100)])
data = pd.concat([pd.DataFrame(x),pd.DataFrame(y),pd.DataFrame(z)],axis=1)
data.columns = ['x','y','z']

# apply method
structure_extraction(data,measure='dcor',repeat_num=10).fig_plot()

#result
(array([-0.75492468,  0.66542939, -0.01750394]),
 array([-0.70624924, -0.69017804, -0.02287991]),
 -0.9648817731111604,
            x         y
 0   0.538123  0.173133
 1   0.524785  0.174132
 2   0.512760  0.176527
 3   0.497919  0.174923
 4   0.480166  0.169195
 ..       ...       ...
 95 -0.216755 -0.483906
 96 -0.222102 -0.502705
 97 -0.222929 -0.515915
 98 -0.215977 -0.519273
 99 -0.214272 -0.529809
 
 [100 rows x 2 columns],
 <Figure size 864x864 with 1 Axes>)
~~~

 ![scatter plot obtained](https://user-images.githubusercontent.com/46952903/194744816-c7842e75-a391-4145-b665-887375b46321.png)

### Parameter
**data: pandas.DataFrame format**  
Target data.  
**measure: {'pearson','kendall','spearman','dcor','ksg','mic','tic','mice','tice','hsic'}, default='ksg'**  
Measure of dependence which estimate linear and nonlinear relationship between 2 variables.  
**opt_method: {'powell','nelder-mead'}, default='powell'**  
Method for solving optimization problem.  
**repeat_num: int, default=10**  
Number of repeat for exploring projection direction. Optimization method may result in a local solution, so we change initial value and repeat procedure several time.  
**aug_threshold: float, default=1e-2**  
Threshold for terminating the procedure in augmented Lagrangian method.  

### Returns
**alpha, beta: array**  
Projection direction such that linear or nonlinear structure appears.  
**dependence: float**  
Value indicated by measure of dependence.  
**proj: pandas.DataFrame**  
Data converted to two-dimensional space.  
**fig: figure**  
Scatter plots obtained.
