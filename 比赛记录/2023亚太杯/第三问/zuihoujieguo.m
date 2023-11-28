clc
clear all
data=readmatrix('数据.xlsx');
shujv=[];
for i=2:size(data,2)
    tmp=myPearson(data(:,1),data(:,i));
    shujv=[shujv;tmp]
end