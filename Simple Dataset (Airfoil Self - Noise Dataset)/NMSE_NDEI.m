%Panteleimon Manouselis AEM:9249
%Function created for Regression (Third) Exercise of Ypologistiki Noimosini
%% Function calculates NMSE and NDEI
function [NMSE,NDEI]=NMSE_NDEI(fis,chkData)
y_hat=evalfis(chkData(:,1:end-1),fis);
y_bar=mean(chkData(:,end));
Se2=sum(((chkData(:,end))-y_hat).^2);
Sx2=sum(((chkData(:,end))-y_bar).^2);
NMSE=Se2/Sx2;
NDEI=sqrt(NMSE);
end