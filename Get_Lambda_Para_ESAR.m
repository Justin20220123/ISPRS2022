function [Polar_Decom_Feat, Polar_Decom_Feat_part]=Get_Lambda_Para_ESAR(path,row,col)

%得到相干矩阵T
% row = 750;
% col = 1024;

% path='E:\20200430Experiment\POLSAR Images\Farm1024-750\T3_with_Polar_Decomp_Para_Bin\';
addpath(path)

tic;
fid = fopen('T11.bin','rb');
T11 = fread(fid,[col,row],'float').'; fclose(fid);
fid = fopen('T22.bin','rb');
T22 = fread(fid,[col,row],'float').'; fclose(fid);
fid = fopen('T33.bin','rb');
T33 = fread(fid,[col,row],'float')'; fclose(fid);
fid = fopen('T12_real.bin','rb');
T12_real = fread(fid,[col,row],'float')'; fclose(fid);
fid = fopen('T12_imag.bin','rb');
T12_imag = fread(fid,[col,row],'float')'; fclose(fid);
T12 = T12_real + i * T12_imag;
fid = fopen('T13_real.bin','rb');
T13_real = fread(fid,[col,row],'float')'; fclose(fid);
fid = fopen('T13_real.bin','rb');
T13_imag = fread(fid,[col,row],'float')'; fclose(fid);
T13 = T13_real + i * T13_imag;
fid = fopen('T23_real.bin','rb');
T23_real = fread(fid,[col,row],'float')'; fclose(fid);
fid = fopen('T23_imag.bin','rb');
T23_imag = fread(fid,[col,row],'float')'; fclose(fid);
T23 = T23_real + i * T23_imag;
toc;
tic;
for ii = 1:row
    for jj = 1:col
        T = [T11(ii,jj) T12(ii,jj) T13(ii,jj);T12(ii,jj)' T22(ii,jj) T23(ii,jj);...
            T13(ii,jj)' T23(ii,jj)' T33(ii,jj)];
        [VT,DT] = eig(T);
        P1 = DT(1,1)/trace(DT);
        P2 = DT(2,2)/trace(DT);
        P3 = DT(3,3)/trace(DT);
        alpha1 = acos(abs(VT(1,1)));
        alpha2 = acos(abs(VT(1,2)));
        alpha3 = acos(abs(VT(1,3)));
        H(ii,jj) = -(P1*log(P1) + P2*log(P2) +P3*log(P3))/log(3);
        alpha(ii,jj) = P1*alpha1 + P2*alpha2 + P3*alpha3;
        A(ii,jj) = (DT(2,2) - DT(1,1))/(DT(2,2) + DT(1,1));
        lambda1(ii,jj)=DT(1,1);
        lambda2(ii,jj)=DT(2,2);
        lambda3(ii,jj)=DT(3,3);
        span(ii,jj)=trace(DT);
        Dspan(ii,jj)=trace(DT)*2*H(ii,jj);
    end
end
toc;
tic;
figure(1);imshow(H);
colormap(jet);colorbar;
title('Polarimetric Scattering Entropy');
figure(2);imshow(alpha);
colormap(jet);colorbar;title('alpha angle');
figure(3);imshow(abs(A));
colormap(jet);colorbar;title('A');
figure(4);imshow(span);
colormap(jet);colorbar;title('SPAN');
figure(5);imshow(Dspan);
colormap(jet);colorbar;title('DSPAN');
toc;
%% matrix2flat
H_flat=reshape(H,1,row*col);
alpha_flat=reshape(alpha,1,row*col);
A_flat=reshape(A,1,row*col);
lambda1_flat=reshape(lambda1,1,row*col);
lambda2_flat=reshape(lambda2,1,row*col);
lambda3_flat=reshape(lambda3,1,row*col);
span_flat=reshape(span,1,row*col);
Dspan_flat=reshape(Dspan,1,row*col);


%% vector2matrix
Polar_Decom_Feat=[lambda1_flat;lambda2_flat;lambda3_flat;span_flat];
Polar_Decom_Feat_part=[lambda1_flat;lambda2_flat;lambda3_flat;span_flat];


end