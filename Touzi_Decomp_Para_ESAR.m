function [DATA_Feat, DATA_Feat_order]=Touzi_Decomp_Para_ESAR(location,r,c)
% % % % clc;
% % % % clear;
% % % % close all;
% % %% number of column and row
% % r=750;
% % c=1024;
%% Find or Given the location of the files of .bin
% location='AIRSAR/C3/';
% location='E:\20200430Experiment\POLSAR Images\Farm1024-750\T3_with_Polar_Decomp_Para_Bin\';

%% Touzi decompose

% obtain alpha1 alpha2 alpha3
file_id = fopen(strcat(location, 'TSVM_alpha_s1.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    ALP1(n,:)=pix;
end
fclose(file_id);

file_id = fopen(strcat(location, 'TSVM_alpha_s2.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    ALP2(n,:)=pix;
end
fclose(file_id);

file_id = fopen(strcat(location, 'TSVM_alpha_s3.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    ALP3(n,:)=pix;
end
fclose(file_id);

file_id = fopen(strcat(location, 'TSVM_alpha_s.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    ALP(n,:)=pix;
end
fclose(file_id);
% obtain phi1 phi2 phi3
file_id = fopen(strcat(location, 'TSVM_phi_s1.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    PHI1(n,:)=pix;
end
fclose(file_id);

file_id = fopen(strcat(location, 'TSVM_phi_s2.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    PHI2(n,:)=pix;
end
fclose(file_id);


file_id = fopen(strcat(location, 'TSVM_phi_s3.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    PHI3(n,:)=pix;
end
fclose(file_id);


file_id = fopen(strcat(location, 'TSVM_phi_s.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    PHI(n,:)=pix;
end
fclose(file_id);

% obtain psi1 psi2 psi3
file_id = fopen(strcat(location, 'TSVM_psi1.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    PSI1(n,:)=pix;
end
fclose(file_id);

file_id = fopen(strcat(location, 'TSVM_psi2.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    PSI2(n,:)=pix;
end
fclose(file_id);


file_id = fopen(strcat(location, 'TSVM_psi3.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    PSI3(n,:)=pix;
end
fclose(file_id);

file_id = fopen(strcat(location, 'TSVM_psi.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    PSI(n,:)=pix;
end
fclose(file_id);


% obtain tau1 tau2  tau3
file_id = fopen(strcat(location, 'TSVM_tau_m1.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    TAU1(n,:)=pix;
end
fclose(file_id);

file_id = fopen(strcat(location, 'TSVM_tau_m2.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    TAU2(n,:)=pix;
end
fclose(file_id);

file_id = fopen(strcat(location, 'TSVM_tau_m3.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    TAU3(n,:)=pix;
end
fclose(file_id);

file_id = fopen(strcat(location, 'TSVM_tau_m.bin'),'rb');
status=fseek(file_id,0,'bof');
for n=1:r
    pix=fread(file_id,c,'float32');
    TAU(n,:)=pix;
end
fclose(file_id);

%% 逐个矩阵转换为大小为r*c的行向量，再拼接为 # of Features * (r*c)的原始数据矩阵

ALP1_DATA=reshape(ALP1,1,r*c);
ALP2_DATA=reshape(ALP2,1,r*c);
ALP3_DATA=reshape(ALP3,1,r*c);
ALP_DATA=reshape(ALP,1,r*c);

PHI1_DATA=reshape(PHI1,1,r*c);
PHI2_DATA=reshape(PHI2,1,r*c);
PHI3_DATA=reshape(PHI3,1,r*c);
PHI_DATA=reshape(PHI,1,r*c);

PSI1_DATA=reshape(PSI1,1,r*c);
PSI2_DATA=reshape(PSI2,1,r*c);
PSI3_DATA=reshape(PSI3,1,r*c);
PSI_DATA=reshape(PSI,1,r*c);

TAU1_DATA=reshape(TAU1,1,r*c);
TAU2_DATA=reshape(TAU2,1,r*c);
TAU3_DATA=reshape(TAU3,1,r*c);
TAU_DATA=reshape(TAU,1,r*c);

DATA_Feat=[ALP1_DATA;ALP2_DATA;ALP3_DATA;PHI1_DATA;PHI2_DATA;PHI3_DATA;
    PSI1_DATA;PSI2_DATA;PSI3_DATA;TAU1_DATA;TAU2_DATA;TAU3_DATA];

DATA_Feat_order=[ALP1_DATA;ALP2_DATA;ALP3_DATA;PHI1_DATA;PHI2_DATA;PHI3_DATA;
    PSI1_DATA;PSI2_DATA;TAU1_DATA;TAU2_DATA;PSI3_DATA;TAU3_DATA];







end

