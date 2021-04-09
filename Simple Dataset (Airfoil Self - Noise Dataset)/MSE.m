%Panteleimon Manouselis AEM:9249
%Function created for Regression (Third)  Exercise of Ypologistiki Noimosini
%% Function calculates the MSE (Mean Square Error)
function cal_mse=MSE(fis,chkData)
output=evalfis(chkData(:,1:end-1),fis);
cal_mse=mse(output,chkData(:,end));
end