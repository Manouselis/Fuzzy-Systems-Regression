%Panteleimon Manouselis AEM:9249
%Function created for Regression (Third)  Exercise of Ypologistiki Noimosini
%% Function calculates Coefficient of Determination (R-Squared)(Sintelestis Prosdiorismou)
function R2=COD(fis,chkData)
output=evalfis(chkData(:,1:end-1),fis);
SSres=sum(((chkData(:,end))-output).^2);
SStot=sum(((chkData(:,end))-(mean(chkData(:,end)))).^2);
R2=1-(SSres/SStot);
end