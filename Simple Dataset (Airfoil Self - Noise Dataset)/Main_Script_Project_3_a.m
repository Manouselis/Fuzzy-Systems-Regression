%Panteleimon Manouselis AEM:9249
%Script created for Regression (Third)  Exercise of Ypologistiki Noimosini
%%
tic
format compact
clear
clc
warning('off','all');

%% Load data
data=load('airfoil_self_noise.dat');
%5 eisodoi, 1 eksodos. 1503 SET dedomenwn
%% Split data
[trnData,chkData,tstData]=split_scale(data,1);
%Splitting and Normalization

%% TSK Models(4)
%TSK 1
opt1 = genfisOptions('GridPartition');
opt1.NumMembershipFunctions = [2 2 2 2 2];
%kai oi pente metavlites eisodou diamerizontai se asafi sinola me plithos
%sinartisewn simetoxis iso me 2

opt1.InputMembershipFunctionType = ["gbellmf" "gbellmf" "gbellmf" "gbellmf" "gbellmf"];
%kai oi pente metavlites eisodou diamerizontai se asafh sinola me
%sinartisis simetoxis oi opoies einai kampanoeideis


opt1.OutputMembershipFunctionType ='constant';
%sinartisi eksodou stathera anti polionimiki
%fis= genfis(inputData,outputData,opt);

fis1=genfis(trnData(:,(1:end-1)),trnData(:,end),opt1);
%fis = genfis(inputData,outputData)

%TSK 2
opt2=genfisOptions('GridPartition');
opt2.NumMembershipFunctions= [3 3 3 3 3];
opt2.InputMembershipFunctionType=["gbellmf" "gbellmf" "gbellmf" "gbellmf" "gbellmf"];
opt2.OutputMembershipFunctionType='constant';

fis2=genfis(trnData(:,(1:end-1)),trnData(:,end),opt2);

%TSK 3
opt3=genfisOptions('GridPartition');
opt3.NumMembershipFunctions= [2 2 2 2 2];
opt3.InputMembershipFunctionType=["gbellmf" "gbellmf" "gbellmf" "gbellmf" "gbellmf"];
opt3.OutputMembershipFunctionType='linear';

fis3=genfis(trnData(:,(1:end-1)),trnData(:,end),opt3);

%TSK 4
opt4=genfisOptions('GridPartition');
opt4.NumMembershipFunctions= [3 3 3 3 3];
opt4.InputMembershipFunctionType=["gbellmf" "gbellmf" "gbellmf" "gbellmf" "gbellmf"];
opt4.OutputMembershipFunctionType='linear';

fis4=genfis(trnData(:,(1:end-1)),trnData(:,end),opt4);
%% Elegxos oti kathe sinartisi simmetoxis exei vathmo epikalispsis >=0.5
figure
plotmf(fis1,'input',1) % gia metavliti eisodou 1 mias kai oles oi metavlites eisodou
%(kai oi 5) exoun tis idies sinartisis simmetoxis ara den exei simasia
%an tha valoume metavliti eisodou 1 h 5
figure
plotmf(fis2,'input',1)
figure
plotmf(fis3,'input',1)
figure
plotmf(fis4,'input',1)

%% Evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%% Training the 4 models
[trnFis1,trnError1,~,valFis1,valError1]=anfis(trnData,fis1,[100 0 0.01 0.9 1.1],[0 0 0 0],chkData);
%anfis(trnData,fis1,[100 0 0.01 0.9 1.1],[],chkData) where 100 is epoch
%number,0 is training goal, 0.01 is the initialStepSize,0.9 is the
%StepSizeDecreaseRate,1.1 is the StepSizeIncreaseRate
%trnFis einai to snapshot tou asafous sistimatos stin teleutaia epoxi
% valFis einai to snapshot tou asafous sistimatos pou proekipse ekeini tin epoxi ekpaideusis opou entopistike to mikrotero sfalma sta chkData
%valError einai to validation error gia kathe epoxi
[trnFis2,trnError2,~,valFis2,valError2]=anfis(trnData,fis2,[100 0 0.01 0.9 1.1],[0 0 0 0],chkData);
toc
tic
[trnFis3,trnError3,~,valFis3,valError3]=anfis(trnData,fis3,[10 0 0.01 0.9 1.1],[0 0 0 0],chkData); % Changed here in order for the code to be executable in the hands of the teacher
[trnFis4,trnError4,~,valFis4,valError4]=anfis(trnData,fis4,[10 0 0.01 0.9 1.1],[0 0 0 0],chkData);

%% Fuzzy set after validation (zitoumeno 1)
MFPlotter(valFis1,size(chkData,2)-1)
MFPlotter(valFis2,size(chkData,2)-1)
MFPlotter(valFis3,size(chkData,2)-1)
MFPlotter(valFis4,size(chkData,2)-1)

%% Learning curves (zitoumeno 2)
LCPlotter(trnError1,valError1);
LCPlotter(trnError2,valError2);
LCPlotter(trnError3,valError3);
LCPlotter(trnError4,valError4);

%% Prediction Error (zitoumeno 3)
%We compare evaluated output of FIS (where input is the test data ) with
%the actual testData output
output1=evalfis(tstData(:,1:end-1),valFis1);
figure
plot(tstData(:,end)-output1,'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for first FIS','Interpreter','Latex');

output2=evalfis(tstData(:,1:end-1),valFis2);
figure
plot(tstData(:,end)-output2,'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for Second FIS','Interpreter','Latex');

output3=evalfis(tstData(:,1:end-1),valFis3);
figure
plot(tstData(:,end)-output3,'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for Third FIS','Interpreter','Latex');

output4=evalfis(tstData(:,1:end-1),valFis4);
figure
plot(tstData(:,end)-output4,'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for Fourth FIS','Interpreter','Latex');

%% (Zitoumeno 4)
% TSK 1
rmse1_test=RMSE(valFis1,tstData);
[nmse1_test,ndei1_test]=NMSE_NDEI(valFis1,tstData);
r2_1_test=COD(valFis1,tstData);


% TSK 2
rmse2_test=RMSE(valFis2,tstData);
[nmse2_test,ndei2_test]=NMSE_NDEI(valFis2,tstData);
r2_2_test=COD(valFis2,tstData);


% TSK 3
rmse3_test=RMSE(valFis3,tstData);
[nmse3_test,ndei3_test]=NMSE_NDEI(valFis3,tstData);
r2_3_test=COD(valFis3,tstData);


% TSK 4
rmse4_test=RMSE(valFis4,tstData);
[nmse4_test,ndei4_test]=NMSE_NDEI(valFis4,tstData);
r2_4_test=COD(valFis2,tstData);
toc