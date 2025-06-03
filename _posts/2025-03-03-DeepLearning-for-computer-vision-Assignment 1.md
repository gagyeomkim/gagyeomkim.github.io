---
title: "[EECS 498-007] Assignment1"

categories: ["EECS 498-007 Deep Learning for Computer Vision"]
tags:
  - ["Deep Learning", "Computer Vision"]

date: 2025-03-03
last_modified_at: 2025-03-03
---

깃허브 : [Github](https://github.com/gagyeomkim/EECS-498-007-Deep-Learning-for-Computer-Vision/tree/main)

## Assignment 1-1: PyTorch 101

해당 과제는 Pytorch를 이용한 튜토리얼이다.
해당 과제는 리뷰보다는, 시간이 날때마다 [깃허브](https://github.com/gagyeomkim/EECS-498-007-Deep-Learning-for-Computer-Vision/blob/main/A1/pytorch101.ipynb)에 있는 전체 코드를 쭉 읽어보는 것이 좋을 것 같다. solution에 대해서만 업로드한다.

```python
import torch


def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from pytorch101.py!')


def create_sample_tensor():
  """
  Return a torch Tensor of shape (3, 2) which is filled with zeros, except for
  element (0, 1) which is set to 10 and element (1, 0) which is set to 100.

  Inputs: None

  Returns:
  - Tensor of shape (3, 2) as described above.
  """
  x = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  x = torch.tensor([[0,10],[100,0],[0,0]])
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return x


def mutate_tensor(x, indices, values):
  """
  Mutate the PyTorch tensor x according to indices and values.
  Specifically, indices is a list [(i0, j0), (i1, j1), ... ] of integer indices,
  and values is a list [v0, v1, ...] of values. This function should mutate x
  by setting:

  x[i0, j0] = v0
  x[i1, j1] = v1

  and so on.

  If the same index pair appears multiple times in indices, you should set x to
  the last one.

  Inputs:
  - x: A Tensor of shape (H, W)
  - indicies: A list of N tuples [(i0, j0), (i1, j1), ..., ]
  - values: A list of N values [v0, v1, ...]

  Returns:
  - The input tensor x
  """
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  for i in range(len(indices)):
    x[indices[i][0], indices[i][1]] = values[i]
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return x


def count_tensor_elements(x):
  """
  Count the number of scalar elements in a tensor x.

  For example, a tensor of shape (10,) has 10 elements.a tensor of shape (3, 4)
  has 12 elements; a tensor of shape (2, 3, 4) has 24 elements, etc.

  You may not use the functions torch.numel or x.numel. The input tensor should
  not be modified.

  Inputs:
  - x: A tensor of any shape

  Returns:
  - num_elements: An integer giving the number of scalar elements in x
  """
  num_elements = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #   You CANNOT use the built-in functions torch.numel(x) or x.numel().      #
  #############################################################################
  # Replace "pass" statement with your code
  shape = 1
  for i in range(x.dim()):
    shape *= x.shape[i]
    
  num_elements = shape
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return num_elements


def create_tensor_of_pi(M, N):
  """
  Returns a Tensor of shape (M, N) filled entirely with the value 3.14

  Inputs:
  - M, N: Positive integers giving the shape of Tensor to create

  Returns:
  - x: A tensor of shape (M, N) filled with the value 3.14
  """
  x = None
  #############################################################################
  #       TODO: Implement this function. It should take one line.             #
  #############################################################################
  x = torch.full((M, N), 3.14)
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return x


def multiples_of_ten(start, stop):
  """
  Returns a Tensor of dtype torch.float64 that contains all of the multiples of
  ten (in order) between start and stop, inclusive. If there are no multiples
  of ten in this range you should return an empty tensor of shape (0,).

  Inputs:
  - start, stop: Integers with start <= stop specifying the range to create.

  Returns:
  - x: Tensor of dtype float64 giving multiples of ten between start and stop.
  """
  assert start <= stop
  x = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  cnt=0 #10이 나온 횟수
  for i in range(start, stop):
    if i != 0 and i % 10 == 0:
        cnt+=1
  if cnt == 0:
        x = torch.zeros((0,), dtype=torch.float64)
  else:
        x = torch.arange(10, 10*cnt+1, 10, dtype=torch.float64)
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return x


def slice_indexing_practice(x):
  """
  Given a two-dimensional tensor x, extract and return several subtensors to
  practice with slice indexing. Each tensor should be created using a single
  slice indexing operation.

  The input tensor should not be modified.

  Input:
  - x: Tensor of shape (M, N) -- M rows, N columns with M >= 3 and N >= 5.

  Returns a tuple of:
  - last_row: Tensor of shape (N,) giving the last row of x. It should be a
    one-dimensional tensor.
  - third_col: Tensor of shape (M, 1) giving the third column of x.
    It should be a two-dimensional tensor.
  - first_two_rows_three_cols: Tensor of shape (2, 3) giving the data in the
    first two rows and first three columns of x.
  - even_rows_odd_cols: Two-dimensional tensor containing the elements in the
    even-valued rows and odd-valued columns of x.
  """
  assert x.shape[0] >= 3
  assert x.shape[1] >= 5
  last_row = None
  third_col = None
  first_two_rows_three_cols = None
  even_rows_odd_cols = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  last_row = x[-1,:]
  third_col = x[:, 2:3]
  first_two_rows_three_cols = x[:2,:3]
  even_rows_odd_cols = x[0::2,1::2]
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  out = (
    last_row,
    third_col,
    first_two_rows_three_cols,
    even_rows_odd_cols,
  )
  return out


def slice_assignment_practice(x):
  """
  Given a two-dimensional tensor of shape (M, N) with M >= 4, N >= 6, mutate its
  first 4 rows and 6 columns so they are equal to:

  [0 1 2 2 2 2]
  [0 1 2 2 2 2]
  [3 4 3 4 5 5]
  [3 4 3 4 5 5]

  Your implementation must obey the following:
  - You should mutate the tensor x in-place and return it
  - You should only modify the first 4 rows and first 6 columns; all other
    elements should remain unchanged
  - You may only mutate the tensor using slice assignment operations, where you
    assign an integer to a slice of the tensor
  - You must use <= 6 slicing operations to achieve the desired result

  Inputs:
  - x: A tensor of shape (M, N) with M >= 4 and N >= 6

  Returns: x
  """
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  x[:2,:6] = torch.tensor([0,1,2,2,2,2])
  x[2:4,:6] = torch.tensor([3,4,3,4,5,5])
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return x


def shuffle_cols(x):
  """
  Re-order the columns of an input tensor as described below.

  Your implementation should construct the output tensor using a single integer
  array indexing operation. The input tensor should not be modified.

  Input:
  - x: A tensor of shape (M, N) with N >= 3

  Returns: A tensor y of shape (M, 4) where:
  - The first two columns of y are copies of the first column of x
  - The third column of y is the same as the third column of x
  - The fourth column of y is the same as the second column of x
  """
  y = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  y = torch.zeros(x.shape[0],4, dtype=torch.int64)
  idx0 = torch.arange(x.shape[0])
  idx1= torch.tensor([0, 0, 2, 1])
  y[:,idx0] = x[:, idx1]
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return y


def reverse_rows(x):
  """
  Reverse the rows of the input tensor.

  Your implementation should construct the output tensor using a single integer
  array indexing operation. The input tensor should not be modified.

  Input:
  - x: A tensor of shape (M, N)

  Returns: A tensor y of shape (M, N) which is the same as x but with the rows
           reversed; that is the first row of y is equal to the last row of x,
           the second row of y is equal to the second to last row of x, etc.
  """
  y = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  y = torch.zeros_like(x)
  idx = torch.arange(x.shape[0])    # [0, 1, ... , M-1]
  idx2 = torch.flip(idx, dims=(0,))
  y[idx, :] = x[idx2, :]
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return y


def take_one_elem_per_col(x):
  """
  Construct a new tensor by picking out one element from each column of the
  input tensor as described below.

  The input tensor should not be modified.

  Input:
  - x: A tensor of shape (M, N) with M >= 4 and N >= 3.

  Returns: A tensor y of shape (3,) such that:
  - The first element of y is the second element of the first column of x
  - The second element of y is the first element of the second column of x
  - The third element of y is the fourth element of the third column of x
  """
  y = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  y = torch.zeros(3, dtype=torch.int64)
  idx_y = torch.arange(3)
  idx2 = [1, 0, 3]
  y[idx_y] = x[idx2, idx_y]
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return y


def count_negative_entries(x):
  """
  Return the number of negative values in the input tensor x.

  Your implementation should perform only a single indexing operation on the
  input tensor. You should not use any explicit loops. The input tensor should
  not be modified.

  Input:
  - x: A tensor of any shape

  Returns:
  - num_neg: Integer giving the number of negative values in x
  """
  num_neg = 0
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  num_neg = torch.sum(x<0)
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return num_neg


def make_one_hot(x):
  """
  Construct a tensor of one-hot-vectors from a list of Python integers.

  Input:
  - x: A list of N integers

  Returns:
  - y: A tensor of shape (N, C) and where C = 1 + max(x) is one more than the max
       value in x. The nth row of y is a one-hot-vector representation of x[n];
       In other words, if x[n] = c then y[n, c] = 1; all other elements of y are
       zeros. The dtype of y should be torch.float32.
  """
  y = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  y = torch.zeros(len(x), 1+max(x))
  idx = torch.arange(len(x))
  y[idx,x] = 1
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return y


def reshape_practice(x):
  """
  Given an input tensor of shape (24,), return a reshaped tensor y of shape
  (3, 8) such that

  y = [
    [x[0], x[1], x[2],  x[3],  x[12], x[13], x[14], x[15]],
    [x[4], x[5], x[6],  x[7],  x[16], x[17], x[18], x[19]],
    [x[8], x[9], x[10], x[11], x[20], x[21], x[22], x[23]],
  ]

  You must construct y by performing a sequence of reshaping operations on x
  (view, t, transpose, permute, contiguous, reshape, etc). The input tensor
  should not be modified.

  Input:
  - x: A tensor of shape (24,)

  Returns:
  - y: A reshaped version of x of shape (3, 8) as described above.
  """
  y = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  x_view = x.view(-1, 4)
  y = torch.zeros(3, 8, dtype=torch.int64)
  y[:,:4] = x_view[:3,:]
  y[:,4:] = x_view[3:,:]
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return y


def zero_row_min(x):
  """
  Return a copy of x, where the minimum value along each row has been set to 0.

  For example, if x is:
  x = torch.tensor([[
        [10, 20, 30],
        [ 2,  5,  1]
      ]])

  Then y = zero_row_min(x) should be:
  torch.tensor([
    [0, 20, 30],
    [2,  5,  0]
  ])

  Your implementation should use reduction and indexing operations; you should
  not use any explicit loops. The input tensor should not be modified.

  Inputs:
  - x: Tensor of shape (M, N)

  Returns:
  - y: Tensor of shape (M, N) that is a copy of x, except the minimum value
       along each row is replaced with 0.
  """
  y = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  y = torch.clone(x)
  idx = torch.argmin(y, dim=1)
  idx_row = torch.arange(x.shape[0])
  y[idx_row, idx] = 0

  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return y


def batched_matrix_multiply(x, y, use_loop=True):
  """
  Perform batched matrix multiplication between the tensor x of shape (B, N, M)
  and the tensor y of shape (B, M, P).

  If use_loop=True, then you should use an explicit loop over the batch
  dimension B. If loop=False, then you should instead compute the batched
  matrix multiply without an explicit loop using a single PyTorch operator.

  Inputs:
  - x: Tensor of shape (B, N, M)
  - y: Tensor of shape (B, M, P)
  - use_loop: Whether to use an explicit Python loop.

  Hint: torch.stack, bmm

  Returns:
  - z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result of
       matrix multiplication between x[i] of shape (N, M) and y[i] of shape
       (M, P). It should have the same dtype as x.
  """
  z = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  if use_loop:
    z = torch.zeros(x.shape[0],x.shape[1],y.shape[2])
    for i in range(x.shape[0]):
        z[i] = torch.mm(x[i],y[i])

  else:
    z = torch.bmm(x, y)

  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return z


def normalize_columns(x):
  """
  Normalize the columns of the matrix x by subtracting the mean and dividing
  by standard deviation of each column. You should return a new tensor; the
  input should not be modified.

  More concretely, given an input tensor x of shape (M, N), produce an output
  tensor y of shape (M, N) where y[i, j] = (x[i, j] - mu_j) / sigma_j, where
  mu_j is the mean of the column x[:, j].

  Your implementation should not use any explicit Python loops (including
  list/set/etc comprehensions); you may only use basic arithmetic operations on
  tensors (+, -, *, /, **, sqrt), the sum reduction function, and reshape
  operations to facilitate broadcasting. You should not use torch.mean,
  torch.std, or their instance method variants x.mean, x.std.

  Input:
  - x: Tensor of shape (M, N).

  Returns:
  - y: Tensor of shape (M, N) as described above. It should have the same dtype
    as the input x.
  """
  y = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  mu_j = torch.sum(x, dim=0) / x.shape[0]
  sigma_j = torch.sqrt(torch.sum((x-mu_j)**2, dim=0) / (x.shape[0] - 1))
  y = (x-mu_j) / sigma_j
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return y


def mm_on_cpu(x, w):
  """
  (helper function) Perform matrix multiplication on CPU.
  PLEASE DO NOT EDIT THIS FUNCTION CALL.

  Input:
  - x: Tensor of shape (A, B), on CPU
  - w: Tensor of shape (B, C), on CPU

  Returns:
  - y: Tensor of shape (A, C) as described above. It should not be in GPU.
  """
  y = x.mm(w)
  return y


def mm_on_gpu(x, w):
  """
  Perform matrix multiplication on GPU

  Specifically, you should (i) place each input on GPU first, and then
  (ii) perform the matrix multiplication operation. Finally, (iii) return the
  final result, which is on CPU for a fair in-place replacement with the mm_on_cpu.

  When you move the tensor to GPU, PLEASE use "your_tensor_intance.cuda()" operation.

  Input:
  - x: Tensor of shape (A, B), on CPU
  - w: Tensor of shape (B, C), on CPU

  Returns:
  - y: Tensor of shape (A, C) as described above. It should not be in GPU.
  """
  y = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  x_gpu = x.cuda()
  w_gpu = w.cuda()
  y = x_gpu.mm(w_gpu)
  y = y.to('cpu')
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return y

```
datatype처럼, `to()`메소드를 이용해서 CUDA를 이용할 수 있다는 것을 챙겨가자. 

또한, 편리한 방식으로 `.cuda()`나 `.cpu()`를 이용할 수 있다.

```python
import torch

if torch.cuda.is_available:
  print('PyTorch can use GPUs!')
else:
  print('PyTorch cannot use GPUs.')
```
```python
# Construct a tensor on the CPU
x0 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print('x0 device:', x0.device)

# Move it to the GPU using .to()
x1 = x0.to('cuda')
print('x1 device:', x1.device)

# Move it to the GPU using .cuda()
x2 = x0.cuda()
print('x2 device:', x2.device)

# Move it back to the CPU using .to()
x3 = x1.to('cpu')
print('x3 device:', x3.device)

# Move it back to the CPU using .cpu()
x4 = x2.cpu()
print('x4 device:', x4.device)

# We can construct tensors directly on the GPU as well
y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64, device='cuda')
print('y device / dtype:', y.device, y.dtype)

# Calling x.to(y) where y is a tensor will return a copy of x with the same
# device and dtype as y
x5 = x0.to(y)
print('x5 device / dtype:', x5.device, x5.dtype)
```
## Assignment 1-2: K-Nearest Neighbors (k-NN)
### Data preprocessing / Visualization
해당 과제에서는 CIFAR-10 dataset을 이용한다.
![fig 1. CIFAR-10](https://i.imgur.com/yIZT5PB.png)
매우 일반적으로 사용되는 dataset이며, 더욱 풍부한 categories를 가지고 있다.

### Subsample the dataset
pytorch에서는 `eecs598.data.cifar10()`이라는 함수를 제공하며, `cifar10`에 대해서 automatically하게 subsampling을 수행할 수 있다. 사용법은 아래와 같이, 인수에 subsampling할  training set과 test set의 수를 넣어주면 된다.
![fig 2. subsampling](https://i.imgur.com/NeSl1pw.png)

### K-Nearest Neighbors (k-NN)
K-Nearest Neighbors를 구현하기 전, k-NN과 관련하여 강의에서 나온 주요 개념들에 대해서 먼저 소개하겠다.
#### 개념
![fig 3. Nearest Neighbor](https://i.imgur.com/dgKfBNL.png)
이 강의에서 처음으로 다룰 Image classifier는 Nearest Neighbor이다.
Learning이라는 말이 무색하게도, train과정에서는 단순히 모든 data와 label에 대해서 저장만 진행한다. 이후, predict에서는 test 이미지와 저장된 train data간에 거리 계산, 유사도 계산등을 통해 비교하고, label을 할당한다.
![fig 4. Distance Metric](https://i.imgur.com/0NbwtCj.png)
similarity 계산에 사용되는 가장 대표적인 방법은 L1과 L2 distance이다.
이 계산에는 보통 Distance Metric을 많이 이용한다.

L1 distance는 위와 같이 각 픽셀값의 절댓값의 차를 구하고, 그 차를 모두 합함으로써 구한다. 즉, 위에 나온 '456'이라는 값이 test image 1개와 training image 1개의 L1 distance인 것이다.

강의 자료에서 제공된 코드를 살펴보면
![fig 5. training data](https://i.imgur.com/8SGlimK.png)
training에서는 단순히, training data x와 그 데이터에 대한 label을 저장한다.
![fig 6. predict](https://i.imgur.com/41yvU0f.png)
이후, test set의 수만큼 반복하면서, 각 test set마다 모든 training set과의 distance를 구하고(여기서는 L1 distance), distance가 가장 낮은 것을 예측 값으로 정하고 있음을 확인할 수 있다.
![fig 7. result](https://i.imgur.com/lSHKhWZ.png)
다음은 CIFAR-10 데이터에 Nearest Neighbors를 적용한 결과이다.
옳은 결과도 있지만, 단지 형태가 비슷한 것을 예측값으로 내놓기도 하는 것을 확인할 수 있다.(4번째 frog와 cat의 색, 구도가 비슷하여, frog를 cat으로 판단하였다.)

![fig 8. decision boundaries](https://i.imgur.com/vNuTvcY.png)
Decision Boundary 개념을 위해 2픽셀 이미지를 가정한 시각화이다.
각 x축과 y축은 픽셀의 intensity(0~255)를 표현한 것이며, 각 점은 개별 train image를 나타낸다. 배경의 색은 새로운 test image가 들어오면 classify를 수행할 decision region space이다. 동그라미 친 부분을 살펴보면, decision boundary는 상당히 noisy한 것을 확인할 수 있다.(이상치데이터 하나 때문에, 초록 영역에 noise가 발생하였다.)

그렇다면 decision boundary를 smoothing하는 전략은 무엇일까?
![fig 9. K-Nearest Neighbors1](https://i.imgur.com/6niZx2A.png)
바로 더 많은 neighbors를 이용하는 것이다.
K=3(neighbors가 3개)일 때, decision boundary가 좀더 smoothing해졌고, outlier의 영향이 줄어듦을 확인할 수 있다.
![fig 10. K-Nearest Neighbors2](https://i.imgur.com/7sANnxA.png)
이처럼 가장 가까운 이웃의 category를 할당한 것이 아니라, 여러개의 이웃에 대해서 majority vote(다수결)방법을 통해 카테고리를 할당하게 된다.
이때, 이웃의 수를 K로 표현하며, K-Nearest Neighbors(k-NN)이라고 부른다.
![fig 10-5. K-Nearest Neighbors3](https://i.imgur.com/xgZEY7u.png)

하지만, K>1이 되면 tie(새로운 data point에 어떤 클래스를 할당해야하는지 불분명해지는 현상, 여기에서는 흰색 부분들)가 발생할 수 있다. 그래서 어떤 카테고리로도 예측하지 못하는 영역이 생기고, 이러한 conflict를 해결하는 방법이 필요하다고 한다.
![fig 11. K-Nearest Neighbors: Distance Metric1](https://i.imgur.com/DgeI6Gj.png)
이미지 similarity 비교를 위해서, L1외에도  L2 distance를 사용할 수 있다.
![fig 12. K-Nearest Neighbors: Distance Metric2](https://i.imgur.com/Gy32FtJ.png)
L1에서는 linear한 부분이 상당히 같은 각도(수평, 수직, 45도..등등)로 나오는데, L2 distance의 경우에는 여전히 linear하지만 다른 방향도 가지는 것을 확인할 수 있다.

> 하지만, 아직 두 distance가 어떤 semantic한 차이를 불러오는지에 대한 직관은 명확하지 않다고 한다.

---

본격적으로, 해당 과제의 목표인 K-Nearest Neighbors(k-NN)을 구현해보자.
아래의 두 라이브러리를 import한 상태를 가정하고, 코드를 설명하겠다.
```python
import torch
import statistics
```

#### Compute distances: Vectorization
> Lets begin with computing the distance matrix between all training and test examples. First we will implement a naive version of the distance computation, using explicit loops over the training and test sets. In the file `knn.py`, implement the function `compute_distances_two_loops`.

첫번째 문제는 training과 test examples에 대해서 distance matrix를 계산해내는 것이다.
- squared Euclidean distances(L2 distance)를 사용하는 조건이 있었다.
2개의 루프 -> vectorization까지 이어지면서, distance matrix를 구현하였다.

```python
def compute_distances_two_loops(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation uses a naive set of nested loops over the training and
  test data.

  The input data may have any number of dimensions -- for example this function
  should be able to compute nearest neighbor between vectors, in which case
  the inputs will have shape (num_{train, test}, D); it should alse be able to
  compute nearest neighbors between images, where the inputs will have shape
  (num_{train, test}, C, H, W). More generally, the inputs will have shape
  (num_{train, test}, D1, D2, ..., Dn); you should flatten each element
  of shape (D1, D2, ..., Dn) into a vector of shape (D1 * D2 * ... * Dn) before
  computing distances.

  The input tensors should not be modified.

  NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
  `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
  You may not use any functions from torch.nn or torch.nn.functional.

  Inputs:
  - x_train: Torch tensor of shape (num_train, D1, D2, ...)
  - x_test: Torch tensor of shape (num_test, D1, D2, ...)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point. It should have the same dtype as x_train.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]  # 500
  num_test = x_test.shape[0]    # 250
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function using a pair of nested loops over the        #
  # training data and the test data.                                           #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  ##############################################################################
  # Replace "pass" statement with your code
  x_train_flatten = x_train.view(num_train, -1)
  x_test_flatten = x_test.view(num_test, -1)
  
  for i in range(num_train):
    for j in range(num_test):
        dists[i, j] = torch.sum(torch.sqrt(torch.abs(x_train_flatten[i, :] - x_test_flatten[j, :])**2))
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists


def compute_distances_one_loop(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation uses only a single loop over the training data.

  Similar to compute_distances_two_loops, this should be able to handle inputs
  with any number of dimensions. The inputs should not be modified.

  NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
  `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
  You may not use any functions from torch.nn or torch.nn.functional.

  Inputs:
  - x_train: Torch tensor of shape (num_train, D1, D2, ...)
  - x_test: Torch tensor of shape (num_test, D1, D2, ...)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function using only a single loop over x_train.       #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  ##############################################################################
  # Replace "pass" statement with your code
  for i in range(num_test):
    x_test_flatten = x_test.view(num_test,-1)   # 각 example마다 전개
    x_train_flatten = x_train.view(num_train,-1)    # 각 example마다 전개
    distance = torch.sum(torch.sqrt(torch.abs(x_test_flatten[i, :] - x_train_flatten)**2), dim=1)
    dists[:,i] = distance.view(1,-1)
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists


def compute_distances_no_loops(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation should not use any Python loops. For memory-efficiency,
  it also should not create any large intermediate tensors; in particular you
  should not create any intermediate tensors with O(num_train*num_test)
  elements.

  Similar to compute_distances_two_loops, this should be able to handle inputs
  with any number of dimensions. The inputs should not be modified.

  NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
  `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
  You may not use any functions from torch.nn or torch.nn.functional.
  Inputs:
  - x_train: Torch tensor of shape (num_train, C, H, W)
  - x_test: Torch tensor of shape (num_test, C, H, W)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function without using any explicit loops and without #
  # creating any intermediate tensors with O(num_train * num_test) elements.   #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  #                                                                            #
  # HINT: Try to formulate the Euclidean distance using two broadcast sums     #
  #       and a matrix multiply.                                               #
  ##############################################################################
  # Replace "pass" statement with your code
  x_train_flatten = x_train.view(num_train, -1)
  x_test_flatten = x_test.view(num_test, -1)

  xx_train_flatten = torch.sum(x_train_flatten.pow(2), dim=1)
  xx_test_flatten = torch.sum(x_test_flatten.pow(2),dim=1)    # 각 example마다 계산.

  x_mm = torch.mm(x_train_flatten, x_test_flatten.t())  # (num_train, num_test)
  
  dists += xx_train_flatten.view(-1,1)
  dists += xx_test_flatten
  dists -= 2*x_mm
  dists = torch.sqrt(dists)
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists

```

`compute_distances_no_loops`를 작성하는 데에 간이 좀 걸렸다. Matrix Multiply와 broadcast를 이용하여 해결하는 방법을 고민해보았는데, 찾은 해결방법은 L2 거리의 식에 포함되는 $(x-y)^2$을 전개하여, $x^2+y^2-2xy$의 식을 이용하는 것이었다.
- $x^2$ = `xx_train_flatten` : training set을 flatten한 후, 행방향으로 제곱한 것의 합을 구함. (num_train, 1)의 shape을 가지게 되며, 각 행마다 제곱값을 가지고 있는 형태로 만듦
- $y^2$ = `xx_test_flatten` : test set을 flatten한 후, 행방향으로 제곱한 것의 합을 구함. (num_test, 1)의 shape을 가지게 되며, 각 행마다 제곱값을 가지고 있는 형태로 만듦
- $xy$ = `x_mm` : test set을 flatten하여 1차원으로 만든 뒤, 이를 `.t()` 함수로 transpose시켜서 training set과 Matrix Multiply를 진행하였다. (num_train, num_test)의 shape을 가지게 된다.

이후, broadcast 기능을 이용하여 distance matrix를 구하는 L2 distance식에 맞게 연산해주면 된다. 
#### Predict labels
```python
from knn import predict_labels

torch.manual_seed(0)
dists = torch.tensor([
    # 1   # 0  # 0
    [0.3, 0.4, 0.1],# 0
    [0.1, 0.5, 0.5],# 1
    [0.4, 0.1, 0.2],# 0
    [0.2, 0.2, 0.4],# 1
    [0.5, 0.3, 0.3],# 2
])
y_train = torch.tensor([0, 1, 0, 1, 2])
y_pred_expected = torch.tensor([1, 0, 0])
y_pred = predict_labels(dists, y_train, k=3)
correct = y_pred.tolist() == y_pred_expected.tolist()
print('Correct: ', correct)
```
이 문제는 풀이 보다 오히려 문제 이해에 시간이 좀 걸렸다..
y_train은 training set에 대한 label,
y_pred_expected는 distance를 이용하여 K-Nearest Neighbors의 방법으로 test set의 label을 predict한 결과이다. 
```python
def predict_labels(dists, y_train, k=1):
  """
  Given distances between all pairs of training and test samples, predict a
  label for each test sample by taking a **majority vote** among its k nearest
  neighbors in the training set.

  In the event of a tie, this function **should** return the smallest label. For
  example, if k=5 and the 5 nearest neighbors to a test example have labels
  [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes), so
  we should return 1 since it is the smallest label.

  This function should not modify any of its inputs.

  Inputs:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  - y_train: Torch tensor of shape (num_train,) giving labels for all training
    samples. Each label is an integer in the range [0, num_classes - 1]
  - k: The number of nearest neighbors to use for classification.

  Returns:
  - y_pred: A torch int64 tensor of shape (num_test,) giving predicted labels
    for the test data, where y_pred[j] is the predicted label for the jth test
    example. Each label should be an integer in the range [0, num_classes - 1].
  """
  num_train, num_test = dists.shape
  y_pred = torch.zeros(num_test, dtype=torch.int64)
  ##############################################################################
  # TODO: Implement this function. You may use an explicit loop over the test  #
  # samples. Hint: Look up the function torch.topk                             #
  ##############################################################################
  # Replace "pass" statement with your code
  for i in range(num_test):
    values, indices = torch.topk(dists[:, i],k=k,dim=0,largest=False)
    output, counts = y_train[indices].unique(return_counts=True)
    y_pred[i] = output[counts.argmax()]
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return y_pred
```

`torch.topk`는 주어진 tensor에 대해서, 인수로 주어진 `dim`을 기준으로 `k`개만큼의 `largest`값(False라면 최솟값)을 찾아준다. 순서대로 내림차순(최솟값인 경우에는 오름차순)으로 찾아준다고 보면 된다.
- return 값으로는 각 value와 index를 반환해준다.각 값의 count를 셀 수 있는게 무엇이 있을지 고민하다가, torch의 `unique`를 찾아볼 수 있었다.

pytorch의 `argmax()`는 최댓값의 인덱스를 반환하는데, 여러개라면 처음으로 나온 최댓값의 index를 반환한다. 따라서, 조건에 따라 예측값을 찾아낼 수 있었다.
```python
class KnnClassifier:
  def __init__(self, x_train, y_train):
    """
    Create a new K-Nearest Neighbor classifier with the specified training data.
    In the initializer we simply memorize the provided training data.

    Inputs:
    - x_train: Torch tensor of shape (num_train, C, H, W) giving training data
    - y_train: int64 torch tensor of shape (num_train,) giving training labels
    """
    ###########################################################################
    # TODO: Implement the initializer for this class. It should perform no    #
    # computation and simply memorize the training data.                      #
    ###########################################################################
    # Replace "pass" statement with your code
    self.xTr = x_train
    self.yTr = y_train
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

  def predict(self, x_test, k=1):
    """
    Make predictions using the classifier.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - k: The number of neighbors to use for predictions

    Returns:
    - y_test_pred: Torch tensor of shape (num_test,) giving predicted labels
      for the test samples.
    """
    y_test_pred = None
    ###########################################################################
    # TODO: Implement this method. You should use the functions you wrote     #
    # above for computing distances (use the no-loop variant) and to predict  #
    # output labels.
    ###########################################################################
    # Replace "pass" statement with your code
    dists = compute_distances_no_loops(self.xTr, x_test)
    y_test_pred = predict_labels(dists, self.yTr, k=k)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_test_pred

  def check_accuracy(self, x_test, y_test, k=1, quiet=False):
    """
    Utility method for checking the accuracy of this classifier on test data.
    Returns the accuracy of the classifier on the test data, and also prints a
    message giving the accuracy.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - y_test: int64 torch tensor of shape (num_test,) giving test labels
    - k: The number of neighbors to use for prediction
    - quiet: If True, don't print a message.

    Returns:
    - accuracy: Accuracy of this classifier on the test data, as a percent.
      Python float in the range [0, 100]
    """
    y_test_pred = self.predict(x_test, k=k)
    num_samples = x_test.shape[0]
    num_correct = (y_test == y_test_pred).sum().item()
    accuracy = 100.0 * num_correct / num_samples
    msg = (f'Got {num_correct} / {num_samples} correct; '
           f'accuracy is {accuracy:.2f}%')
    if not quiet:
      print(msg)
    return accuracy
```
`KnnClassifier`를 구현하는 과제였는데, `__init__`에선 단순히 training data를 저장하고, predict 부분에서는 이전에 구한 distance 함수와 predict label 함수를 인수 위치에 주의하여 적절히 조합하면 되었다.
이를 통해 test set에 대해서, `y_test_pred`(test data의 label값)을 구할 수 있다.

#### Cross-validation
cross-validation을 구현하기 전에, 간단하게 개념을 살펴보자.
![fig 13. hyperparameter](https://i.imgur.com/BAdILJB.png)

hyperparameter는 training data로부터 학습할 수 있는 것이 아니라, 사람이 직접 설정해야하는 파라미터이다. 학습이 진행됨에 따라서, 어떤 값이 가장 잘 맞는지 손수 찾아봐야한다. 

이러한 하이퍼 파라미터를 세팅하기 위해서 몇가지 전략이 존재한다.

![fig 14. validation set](https://i.imgur.com/4T2WFBY.png)

1.training data의 accuracy가 가장 높은 하이퍼 파라미터를 선택하는 것

이러한 접근은 절대 하면 안된다. k-nn을 예시로 들었을 때, k=1이라고 하는 것과 같은데, 이 경우에는 knn classifier는 training data에 대해선 100%의 accuracy를 보이기 때문에, 범용 능력을 가지는지 확인할 수 없다.

2.training data와 test set 두 요소로 분리하는 것

두번째 아이디어는 training set과 test set 두 요소로 분리하는 것이다. 첫번째 데이터에 비해 unseen data에 대한 성능을 확인(범용성을 확인)할 수 있지만, hyper parameter settting에서 test set에 대해 적합한 파라미터를 가지게 되며, 이 경우, test set은 더이상 unseen data로 유지되지 않는다. 즉, 모델의 성능을 평가할 unseen data가 반드시 필요하다.

3.training set과 validation set, test set으로 분리하는 것

세번째 아이디어를 이용해야한다. 여기서 evaluation(hyper parameter setting)은 validation set을 통해 진행하고, 이를 통해 하이퍼 파라미터를 업데이트 하면서 test set에 대한 unseen 상태를 유지해야한다. test set은 최종단계에서 오직 1번만 실행한다.

![fig 15. cross-validation se](https://i.imgur.com/4f1sBzW.png)
Cross-Validation은 데이터 셋을 fold라 불리는 덩어리(chunk)로 나누고, 나눠진 fold가 5개라면 5번의 iteration을 돌면서 각 iteration마다 validation set으로 이용하는 fold를 변경하는 것이다.(이 경우에도 test set은 건들면 안된다.) 이 방법은 이론상 제일 좋은 방법이지만, 연산량이 매우 크기 때문에 적용하기 힘든 경우가 많다.

우리는 세번째 방법을 이용해서, 과제를 해결한다.
```python
def knn_cross_validate(x_train, y_train, num_folds=5, k_choices=None):
  """
  Perform cross-validation for KnnClassifier.

  Inputs:
  - x_train: Tensor of shape (num_train, C, H, W) giving all training data
  - y_train: int64 tensor of shape (num_train,) giving labels for training data
  - num_folds: Integer giving the number of folds to use
  - k_choices: List of integers giving the values of k to try

  Returns:
  - k_to_accuracies: Dictionary mapping values of k to lists, where
    k_to_accuracies[k][i] is the accuracy on the ith fold of a KnnClassifier
    that uses k nearest neighbors.
  """
  if k_choices is None:
    # Use default values
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

  # First we divide the training data into num_folds equally-sized folds.
  x_train_folds = []
  y_train_folds = []
  ##############################################################################
  # TODO: Split the training data and images into folds. After splitting,      #
  # x_train_folds and y_train_folds should be lists of length num_folds, where #
  # y_train_folds[i] is the label vector for images in x_train_folds[i].       #
  # Hint: torch.chunk                                                          #
  ##############################################################################
  # Replace "pass" statement with your code
  x_train_folds = torch.chunk(x_train, num_folds)
  y_train_folds = torch.chunk(y_train, num_folds)
  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################

  # A dictionary holding the accuracies for different values of k that we find
  # when running cross-validation. After running cross-validation,
  # k_to_accuracies[k] should be a list of length num_folds giving the different
  # accuracies we found when trying KnnClassifiers that use k neighbors.
  k_to_accuracies = {}

  ##############################################################################
  # TODO: Perform cross-validation to find the best value of k. For each value #
  # of k in k_choices, run the k-nearest-neighbor algorithm num_folds times;   #
  # in each case you'll use all but one fold as training data, and use the     #
  # last fold as a validation set. Store the accuracies for all folds and all  #
  # values in k in k_to_accuracies.   HINT: torch.cat                          #
  ##############################################################################
  # Replace "pass" statement with your code
  for k in k_choices:
    acc_list = []
    for i in range(num_folds):
        if i == 0:
            x_train = torch.cat((x_train_folds[1:]))
            y_train = torch.cat((y_train_folds[1:]))
        elif i!= num_folds-1:
            x_train = torch.cat((torch.cat(x_train_folds[:i]),torch.cat(x_train_folds[i+1:])))
            y_train = torch.cat((torch.cat(y_train_folds[:i]),torch.cat(y_train_folds[i+1:])))
        else:
            x_train = torch.cat((x_train_folds[:i]))
            y_train = torch.cat((y_train_folds[:i]))
        x_test = x_train_folds[i]
        y_test = y_train_folds[i]

        classifier = KnnClassifier(x_train, y_train)
        acc_list.append(classifier.check_accuracy(x_test, y_test, k=k))
    k_to_accuracies[k] = acc_list
  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################

  return k_to_accuracies

```
`torch.chunk` : 인수만큼 해당 tensor를 분할해줌
`torch.cat` : tensor를 이어붙여줌
- `torch.cat`에서 괄호를 주의해야한다. dim이 들어가는 부분이 있기 때문에, 각 tensor를 `()`로 한번 더 감싸줘야하는 것으로 이해했다.

아래는 가장 적절한 k값을 찾는 코드이다. 반복문을 돌며 최솟값을 찾는 로직을 사용하였다.
```python
def knn_get_best_k(k_to_accuracies):
  """
  Select the best value for k, from the cross-validation result from
  knn_cross_validate. If there are multiple k's available, then you SHOULD
  choose the smallest k among all possible answer.

  Inputs:
  - k_to_accuracies: Dictionary mapping values of k to lists, where
    k_to_accuracies[k][i] is the accuracy on the ith fold of a KnnClassifier
    that uses k nearest neighbors.

  Returns:
  - best_k: best (and smallest if there is a conflict) k value based on
            the k_to_accuracies info
  """
  best_k = 0
  ##############################################################################
  # TODO: Use the results of cross-validation stored in k_to_accuracies to     #
  # choose the value of k, and store the result in best_k. You should choose   #
  # the value of k that has the highest mean accuracy accross all folds.       #
  ##############################################################################
  # Replace "pass" statement with your code
  best_mean = 0
  for k in k_to_accuracies.keys():
    m = torch.tensor(k_to_accuracies[k]).mean()
    if m > best_mean:
        best_mean = m
        best_k = k
  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################
  return best_k
```

## Plus&ref
추가로 아래 게시글에서 training sample이 많아질 경우, k-nn을 사용하였을 때 발생하는  Curse of Dimensionality를 살펴보면 좋을 것 같다.

![Curse of Dimensionality](https://i.imgur.com/55Mtnqv.png)
k-NN의 표현력은 training sample이 많아질수록 좋아지는데, 이는 Curse of Dimensionality 현상을 일으킬 수 있다. feature space의 dimension이 uniform하게 증가할 때, 같은 밀도(coverage)를 유지하기 위해서는 데이터의 샘플이 지수적으로 필요하다는 내용이다.

정리를 아주 잘해주셔서 유용하게 사용할 수 있었다.

<https://ploradoaa.tistory.com/76>