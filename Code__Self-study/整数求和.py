
print("我来帮你算从1到n的所有整数的和")
print("告诉我你想计算的n")
n=int(input())
counter=1
sum=0
while counter<=n:
    sum=sum+counter
    counter+=1
print("从1到%d的和为: %d" %(n,sum))

