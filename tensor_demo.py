import torch
import numpy as np

print("--------tensor create:--------")
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

print(f"x_data Tensor: \n {x_data} \n")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"x_np Tensor: \n {x_np} \n")


x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


print("--------tensor attributes:--------")
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


#  indexing and slicing:
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)


# Joining tensors 连接张量
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# dim是维度，在这里只有二维，dim=0表示行，dim=1表示列
print(t1)


# ----------------Arithmetic operations--------------------------------
print("--------Arithmetic operations:--------")
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
print(y1)
y2 = tensor.matmul(tensor.T)
print(y2)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
print(z1)
z2 = tensor.mul(tensor)
print(z2)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z3)
