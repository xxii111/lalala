function [r1,cd1,peaksnr1,r2,cd2,peaksnr2,r3,cd3,peaksnr3]=rdvalue2(origin_data,data,M,N)
len=size(data,1);

%基本层无损
data_e=zeros(len,2);
data_2=zeros(len,2);
data_3=zeros(len,2);
d1=zeros(len,2);
d2=zeros(len,1);

%对（X，Y）量化
[idx1,center_e]=kmeans(data,M);
data_e(1:len,:)=center_e(idx1(1:len),:);

%对Y量化
[idx2,center_y]=kmeans(data(:,2),M);
data_y(1:len)=center_y(idx2(1:len));

%计算无基本层预测时的mse1
d1(1:len,:)=data_e(1:len,:)-data(1:len,:);
mse1=mse(d1);
peaksnr1=10*log10((8^2)/mse1);
cd1 = classdistortion(data,data_e,M);

%计算无基本层预测时的rate1
distribution1=CreateDistribution(data_e(:,1)',data_e(:,2)');
r1=JointEntropy(distribution1);

%计算有基本层预测时的mse2
d2(1:len)=data_y(1:len)'-data(1:len,2);
mse2=mse(d2);
peaksnr2=10*log10((8^2)/mse2);
data_2(:,1)=data(:,1);
data_2(:,2)=data_y;
cd2 = classdistortion(data,data_2,M);

%计算有基本层预测时的rate2
distribution2=CreateDistributionX(data_y);
r2=Entropy(distribution2);

%基本层有损 
%对基本层有损压缩
data_b=zeros(len,1);
residual=zeros(len,2);
d3=zeros(len,2);
data_d=zeros(len,2);

%对X量化
[idx3,center_b]=kmeans(data(:,1),N);
data_b(1:len)=center_b(idx3(1:len));

%计算(X-Xb`,Y)=residual
residual(1:len,1)=data(1:len,1)-data_b(1:len);
residual(1:len,2)=data(1:len,2);

%对(X-Xb`,Y)编码
[idx4,center_l]=kmeans(residual,M);
data_d(1:len,:)=center_l(idx4(1:len),:);

%计算有基本层时的mse3
d3(1:len,:)=data_d(1:len,:)-residual(1:len,:);
mse3=mse(d3);
peaksnr3=10*log10((8^2)/mse3);
data_3(:,1)=data_d(:,1)+data(:,1);
data_3(:,2)=data_d(:,2);
cd3 = classdistortion(data,data_3,M);
%计算有基本层时的rate3
distribution3=CreateDistribution(data_d(:,1)',data_d(:,2)');
r3=JointEntropy(distribution3);

% % 聚类中心可视化
figure(M+100)
plot(origin_data(1:140,1),origin_data(1:140,2),'b.');
hold on;
plot(origin_data(141:280,1),origin_data(141:280,2),'r.');
plot(origin_data(281:350,1),origin_data(281:350,2),'g.');
plot(origin_data(351:490,1),origin_data(351:490,2),'y.');
plot(origin_data(491:560,1),origin_data(491:560,2),'m.');
plot(origin_data(561:700,1),origin_data(561:700,2),'c.');
plot(center_e(:,1),center_e(:,2),'k*','MarkerSize',10);
legend("Z_1","Z_2","Z_3","Z_4","Z_5","Z_6","聚类中心")
title('聚类中心可视化示意图')

