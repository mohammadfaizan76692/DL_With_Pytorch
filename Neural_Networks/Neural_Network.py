import torch

tensor1 = torch.rand(2,3)
tensor2 = torch.rand(3,2)

## matrix multiplication
print(tensor1@tensor2)


## Batch matrix multiplication (important)
batch = 32
n= 100
m = 40
p = 10
tensor1 = torch.rand(batch, n, m)
tensor2  = torch.rand(batch,m,p)
result = torch.bmm(tensor1, tensor2)
print(result.shape)

## Broad Casting
tensor1 = torch.rand(5,5)
tensor2 = torch.arange(5) # shape 1, 5 --> [0,1,2,3,4] 
## after broadcastering (5,5) to perform subtraction 
"""
[[0,1,2,3,4],
[0,1,2,3,4],
[0,1,2,3,4],
[0,1,2,3,4],
[0,1,2,3,4]]
"""
print(tensor2)
print(tensor1)
print(tensor1-tensor2)

tensor1 = torch.tensor([[1,2,3],[5,4,6],[7,8,9],[10,11,12],[13,15,14]])
sum_tensor1 = torch.sum(tensor1, axis =0) ## column wali axis , 5 column 5 ans
sum_tensor2 = torch.sum(tensor1,axis =1)  ## row axis  wali axis , 3 row 3 ans
print(sum_tensor1)
print(sum_tensor2)
print(tensor1)
result , indexes = torch.max(tensor1,axis=0)
print(result)
print(indexes)
result , indexes = torch.min(tensor1,axis=1)
print(result)
print(indexes)
# just getting indexes  
indexes = torch.argmax(tensor1, axis = 0)
print(indexes)

tensor1 = torch.rand((2,3,4))
sorted_result, indeces = torch.sort(tensor1,dim =0) ## Value sorted along that dimension
print(indeces)
print(sorted_result)