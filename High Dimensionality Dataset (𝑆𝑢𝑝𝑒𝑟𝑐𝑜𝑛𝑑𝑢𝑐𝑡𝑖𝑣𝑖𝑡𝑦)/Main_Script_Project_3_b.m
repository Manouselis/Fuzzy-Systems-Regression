%Panteleimon Manouselis AEM:9249
%Script created for Regression (Third) Exercise of Ypologistiki Noimosini
%%
tic
format compact
clear
clc
warning('off','all');
%suppresing warning outputs

%% Load data
data=csvread('train.csv',1,0);
%21263 SET dedomenwn
%% Split data
[trnData,chkData,tstData]=split_scale(data,1);
%Splitting and Normalization

Num_Feat = [3 7 13 21]; % number of features

radii = [0.2 0.4 0.6 0.8 0.9]; % values for radii


%% GRID SEARCH & 5-fold cross validation

[idx, weights] = relieff(data(:, 1:end - 1), data(:, end), 100);
%Similar to ReliefF, RReliefF also penalizes the predictors that give different values to neighbors with the same response values, and rewards predictors that give different values to neighbors with different response values.
%However, RReliefF uses intermediate weights to compute the final predictor weights.
bar(weights(idx))
xlabel('Predictor rank')
ylabel('Predictor importance weight')
% katatasoume tis stiles se seira simantikotitas me vasi to varos tous

for i = 1 : length(Num_Feat)
    
    for j = 1 : length(radii)
        %dokimazoume olous tous sindiasmous diaforetikwn arithmon apo
        %features kai diaforetikon arithmon apo radii
        
        parti_data = cvpartition(trnData(:, end), 'KFold', 5);
        %Epilegetai h  stili apo to trnData mas kai diaxwrizetai
        %se dio sinola dedomenon (80% kai 20% tou arxikou sinolou
        %dedomenwn). Ayto ginetai 5 fores (5 set diaxorismenwn dedomenwn)
        
        %Creating the initial FIS which later changes because of cross
        %Validation
        opt = genfisOptions('SubtractiveClustering');
        opt.ClusterInfluenceRange=radii(j);
        init_fis =genfis(trnData(:, idx(1:Num_Feat(i))), trnData(:, end), opt);
        Rule_Grid(i, j) = length(init_fis.rule);
        if (Rule_Grid(i, j) == 1 || Rule_Grid(i,j) > 200)
            continue;
        end
        % if only one rule exists we cannot create FIS
        %if there are more than 200 rules then the computational time is
        %very large an thus we choose not to create the FIS
        
        %% 5-fold cross Validation
        fprintf('\n Number of features %d\n', Num_Feat(i));
        fprintf('\n Radii is equal to %d\n', radii(j));
        
        %%%%%%
        %Cross validation happens here
        error=CV(init_fis,trnData,chkData,parti_data,Num_Feat,i,idx);
        %%%%%%
        
        Tot_CV_error=sum(error);
        Average_CV_error=Tot_CV_error/5;
        Mean_error(i,j)=Average_CV_error/length(chkData(:,end));
    end
end
toc
%Rule_Grid
%% Plotter (Erotima 2)

for i=1:length(Num_Feat)
    for j=1:length(radii)
        if(Mean_error(i,j)==0)
            Mean_error(i,j)=nan;
            %removing errors that are zero since they are fake (in reality
            %they are not zero)
        end
    end
end
[mini,inde1]=min(Mean_error,[],'omitnan');%returns a vector with the indices of the smallest values of each column of A( the NaN are ommited)
[mini,inde2]=min(mini);%we find the indice of the smallest value

figure
stem3(radii,Num_Feat,Mean_error,'filled')
hold on
stem3(radii(inde2),Num_Feat(inde1(inde2)),Mean_error(inde1(inde2),inde2),'r','filled'); %we plot the smallest value found differently than the other values
%for that we use the indices we have found above
grid on;
xlabel('Radius','Interpreter','Latex');
ylabel('Number of Features','Interpreter','Latex');
zlabel('Mean Error','Interpreter','Latex');

% Diagramma opou apeikonizetai to sfalma se sxesi me tin aktina gia kathe
% arithmo xaraktiristikwn
figure
plot(radii,Mean_error(1,:),'-or',radii,Mean_error(2,:),'--ob',radii,Mean_error(3,:),':om',radii,Mean_error(4,:),'-.og')
legend(['Number of Features is ', num2str(Num_Feat(1))],['Number of Features is ',num2str(Num_Feat(2))],['Number of Features is ',num2str(Num_Feat(3))],['Number of Features is ',num2str(Num_Feat(4))],'Interpeter','Latex')
xlabel('Radius','Interpreter','Latex');
ylabel('Mean Error','Interpreter','Latex');

% Diagramma opou apeikonizetai to sfalma se sxesi me ton arithmo
% xaraktiristikwn gia kathe aktina
figure
plot(Num_Feat,Mean_error(:,1),'-or',Num_Feat,Mean_error(:,2),'--ob',Num_Feat,Mean_error(:,3),':om',Num_Feat,Mean_error(:,4),'-.og',Num_Feat,Mean_error(:,5),'-.oc')
legend(['Radii is ', num2str(radii(1))],['Radii is ',num2str(radii(2))],['Radii is ',num2str(radii(3))],['Radii is ',num2str(radii(4))],['Radii is ',num2str(radii(5))],'Interpeter','Latex')
xlabel('Number of Features','Interpreter','Latex');
ylabel('Mean Error','Interpreter','Latex');

% Diagramma opou apeikonizetai to sfalma se sxesi me ton arithmo ton
% kanonwn
figure
stem(Rule_Grid,Mean_error,'filled')

xlabel('Number of Rules','Interpreter','Latex');
ylabel('Mean Error','Interpreter','Latex');
xlim([0 25]);

%% Final TSK Model Training
%Minimum Error when we have 21 features and 0.4 radius
opt_final=genfisOptions('SubtractiveClustering');
opt_final.ClusterInfluenceRange=radii(2);% radius=0.4
final_fis=genfis(trnData(:,idx(1:21)),trnData(:,end),opt_final);
final_trnData=[trnData(:,idx(1:21)) trnData(:,end)];
final_chkData=[chkData(:,idx(1:21)) chkData(:,end)];
[trnFisfin,trnErrorfin,~,valFisfin,valErrorfin]=anfis(final_trnData,final_fis,[100 0 0.01 0.9 1.1],[0 0 0 0],final_chkData);
%Training our model for 100 epochs. Xrisimopoioumai tis veltistes times ton
%parametrwn. Gia auto oi stiles tou trnData kai chkData pou tha
%xrisimopoieithoun epilegontai me vasi to idx(1:21) dedomenou oti theloume
%21 xaraktiristika (21 stiles )

%% Predictions of trained model and real values (zitoumeno 1 erotima 3)
y_hat=evalfis(valFisfin,tstData(:,idx(1:21)));
y_actual=tstData(:,end);
error=y_actual-y_hat;

figure
stem(y_hat)
hold on
stem(y_actual)
hold off
grid on;
legend('Actual Output','Predicted Output','Interpeter','Latex')
xlabel('Sample Size','Interpreter','Latex');
ylabel('Magnitude of Output','Interpreter','Latex');

figure
stem(error,'LineWidth',2);
grid on;
xlabel('Sample Size','Interpreter','Latex');
ylabel('Prediction Error for final FIS','Interpreter','Latex');

%% Learning Curves (zitoumeno 2 erotima 3)
LCPlotter(trnErrorfin,valErrorfin)
ylim([12 18]);
title('');
%% Fuzzy set initial and afterwards figure (zitoumeno 3 erotima 3)
dimension=size(final_chkData,2)-1;
rand_input_fis=randperm(dimension,5);
%returns a row vector containing k unique integers selected randomly from 1 to n.

%Before the training
figure
for i=1:5
    subplot(2,3,i)
    hold on
    plotmf(final_fis,'input',rand_input_fis(i))
    ylabel('Degree of membership before training the final model', 'Interpreter', 'latex')
    title(['Membership function for number ',num2str(rand_input_fis(i)), ' input of FIS'],  'Interpreter', 'latex')
end

%After the training
figure
for i=1:5
    subplot(2,3,i)
    hold on
    plotmf(valFisfin,'input',rand_input_fis(i))
    ylabel('Degree of membership after training the final model', 'Interpreter', 'latex')
    title(['Membership function for number ',num2str(rand_input_fis(i)), ' input of FIS'],  'Interpreter', 'latex')
end

%% RMSE,NMSE,NDEI,R^2 (zitoumeno 4 erotima 3)
rmseFin=RMSE(valFisfin,[tstData(:,idx(1:21)) tstData(:,end)]);
[nmseFin,ndeiFin]=NMSE_NDEI(valFisfin,[tstData(:,idx(1:21)) tstData(:,end)]);
r2Fin=COD(valFisfin,[tstData(:,idx(1:21)) tstData(:,end)]);