clc
clear ;
close all ;
%% according to visually standard image getting ��ȡ����ʾ���ͼ����������
image=imread('E:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\standard image\��׼ͼ.bmp');
figure(),imshow(uint8(image))
[r,c,gg]=size(image);

%%
addpath 'E:\20200430Experiment\AE_with_finetune\';
[val, Numval,indexing] =Findvalue(image); %�ҵ�ͼ���У�����ֵ����𣬸�����Ϣ����������indexing
K=size(indexing,1);%�������
Labels=Visual2Labels(image,indexing);%���ӻ������ͼ���������indexingת��Ϊ��ǩ����Labels

%% ����Ӧ�õĸ������ݵ�������Ϣ���������ݾ���


for iii=3:2:11
    
    path=strcat('E:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\LeeWindowTouWind3\',int2str(iii),'\T3\');
    
    %% Obtain All Parameters of Touzi_Decomposition
    [DATA_Feat, DATA_Feat_order]=Touzi_Decomp_ALLPara_ESAR(path,r,c);
      
    %% Paras.
    DATA=DATA_Feat;
    
    [Dim,~]=size(DATA);
    inputSize =Dim;              % Inputlayer
    numClasses = K-1;           % Outputlayer  Layer 3
    softmaxLambda = 1e-8;
    alpha=0.01;
 
    Ntimes=30;
    
    %% demo function1
    
    DATA1=DATA;      % ensure the data invariable
    addpath 'E:\20200430Experiment\data_preprocessing\'
    DATA1=ZCA_Whitening(DATA1);
    [Dim1,~]=size(DATA1);
    
    addpath 'E:\20200430Experiment\'
    for nn=1:Dim1
        for kkk=1:Ntimes
            [pred1ZA(kkk,:),acc1ZA(kkk,nn)]=Softmax_Classifier_JustinESAR(nn,numClasses,softmaxLambda,DATA1(1:nn,:),Labels,K,alpha);
            nn
            disp('����ַ�����!!!!!!**********************************\n%%%%$$$$$$$$')
        end
    end
   accAA(iii,:) =mean(acc1ZA);

    %% demo function2
    DATA_part= DATA_Feat_order;
    
    DATA2=DATA_part;
    DATA2=ZCA_Whitening(DATA2);
    [Dim2,~]=size(DATA2);
    
    for nn=1:Dim2
        for kkk=1:Ntimes
            [pred1ZP(kkk,:),acc1ZP(kkk,nn)]=Softmax_Classifier_JustinESAR(nn,numClasses,softmaxLambda,DATA2(1:nn,:),Labels,K,alpha);
            
            nn
            disp('����ַ�����!!!!!!**********************************\n%%%%$$$$$$$$')
        end
    end
    
     accAB(iii,:) =mean(acc1ZP);

end



%% ������֮ǰ�Ĳ����Ĳ�ͬ�Ĳ���
figure()
plot(1:Dim1,accAA(3,:),'-ro',1:Dim1,accAA(5,:),'-g+',1:Dim1,accAA(7,:),'-b*')
hold on
plot(1:Dim1,accAA(9,:),'-ks',1:Dim1,accAA(11,:),'-md')

x1=1:1:Dim1;
cell_1={'\lambda_1','\lambda_2','\lambda_3','SPAN',...
    '\alpha_{s1}','\alpha_{s2}','\alpha_{s3}','\Phi_{\alpha_{s1}}', '\Phi_{\alpha_{s2}}', '\Phi_{\alpha_{s3}}', ...
    '\Psi_1','\Psi_2','\Psi_3', '\tau_1','\tau_2','\tau_3'};%���෽��
set(gca,'xtick',x1);
set(gca,'xticklabel',cell_1);

xlabel('Touzi Parameters');
ylabel('OA');

axis([1 Dim1 0.1 1.0])     % ʹ������Ŀ̶Ⱦ��ȷֲ���X���ϣ��������ұ�����������
set(gca,'FontSize',10);    % ��������������ִ�С������legend���ִ�С��
legend('size=3','size=5','size=7','size=9','size=11','Location','SouthEast')  % ���������͵�˵����Ϣ��λ�ã����½�
xtickangle(-45)%  controll the rotation angle of xlabel
ytickangle(-45)%  controll the rotation angle of ylabel



%%
figure()
plot(1:Dim2,accAB(3,:),'-ro',1:Dim2,accAB(5,:),'-g+',1:Dim2,accAB(7,:),'-b*')
hold on
plot(1:Dim2,accAB(9,:),'-ks',1:Dim2,accAB(11,:),'-md')

x2=1:1:Dim2;
cell_2={'\lambda_1','\lambda_2','\lambda_3','SPAN',  ...
    '\alpha_{s1}','\alpha_{s2}','\alpha_{s3}','\Phi_\alpha_{s1}', '\Phi_\alpha_{s2}', '\Phi_\alpha_{s3}',...
    '\Psi_1','\Psi_2','\tau_1','\tau_2','\Psi_3','\tau_3'};%���෽��
set(gca,'xtick',x2);
set(gca,'xticklabel',cell_2);
xlabel('Touzi Parameters');
ylabel('OA');
axis([1 Dim2 0.2 0.9])     % ʹ������Ŀ̶Ⱦ��ȷֲ���X���ϣ��������ұ�����������
set(gca,'FontSize',18);    % ��������������ִ�С������legend���ִ�С��
legend('size=3','size=5','size=7','size=9','size=11','Location','SouthEast')  % ���������͵�˵����Ϣ��λ�ã����½�
xtickangle(-90)%  controll the rotation angle of xlabel
ytickangle(-45)%  controll the rotation angle of ylabel

saveas(gcf,'ESARLeeWindTouWind3.bmp')
%%

% save ESARLeeWindTouWind3.mat



% % % % % % % % % % % % % % % % % %% ���ѡȡ������ά�ȵ������������������Ծ��ȵ�Ӱ�� #Features =11; չʾ��� Average_OA=0.7010
% % % % % % % % % % % % % % % % % num=11; %huozhe >11
% % % % % % % % % % % % % % % % % DATA3=DATA;      % ensure the data invariable
% % % % % % % % % % % % % % % % % addpath 'E:\20200430Experiment\data_preprocessing\'
% % % % % % % % % % % % % % % % % DATA3=ZCA_Whitening(DATA3);
% % % % % % % % % % % % % % % % % [Dim3,~]=size(DATA3);
% % % % % % % % % % % % % % % % % uu=randperm(Dim3);
% % % % % % % % % % % % % % % % % vv=uu(:,1:num);
% % % % % % % % % % % % % % % % % DATA4=DATA3(vv,:);
% % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % for kkk=1:Ntimes
% % % % % % % % % % % % % % % % %     [pred11_RZP(kkk,:),acc11_RZP(kkk)]=Softmax_Classifier_Justin(num,numClasses,softmaxLambda,DATA4,Labels,K);
% % % % % % % % % % % % % % % % %     disp('�����������˳��ѡȡ���������������Ľ��\n%%%%$$$$$$$$')
% % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % %%  #Features =11 �����Ǹ���ͼ���������Ϣ��ֱ��ͼ����ѡȡ������#Features=11;չʾ��� Average_OA=0.8615
% % % % % % % % % % % % % % % % % % % % % best_2=15;
% % % % % % % % % % % % % % % % % DATA5=[DATA2(1:9,:);DATA2(14:15,:)];
% % % % % % % % % % % % % % % % % [best_2,~]=size(DATA5);
% % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % for kkk=1:Ntimes
% % % % % % % % % % % % % % % % %     [pred11_SZP(kkk,:),acc11_SZP(kkk)]=Softmax_Classifier_Justin(best_2,numClasses,softmaxLambda,DATA5,Labels,K);
% % % % % % % % % % % % % % % % %     disp('�������ֱ��ͼ����������ͼ���ɵ�����˳��ѡȡ���������������Ľ��\n%%%%$$$$$$$$')
% % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % %%  �����е�����ֵ����ZCAԤ����Ȼ�����ֱ�ӷ���;չʾ��� 0.8773
% % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % for kkk=1:Ntimes
% % % % % % % % % % % % % % % % %     [pred11_AZP(kkk,:),acc11_AZP(kkk)]=Softmax_Classifier_Justin(Dim,numClasses,softmaxLambda,DATA1,Labels,K);
% % % % % % % % % % % % % % % % %     disp('������������Ľ��\n%%%%$$$$$$$$')
% % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % %% ��ʾ���ѡȡ���Ų���������������ѡ��ķ������������Ų������Լ����еĲ�����������ƽ��
% % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % figure()
% % % % % % % % % % % % % % % % % plot(1:Ntimes,acc11_RZP,'-r*',1:Ntimes,acc11_SZP,'-gd',1:Ntimes,acc11_AZP,'-bo')
% % % % % % % % % % % % % % % % % hold on
% % % % % % % % % % % % % % % % % plot(1:Ntimes,repmat(mean(acc11_RZP),1,Ntimes),'-.r',1:Ntimes,repmat(mean(acc11_SZP),1,Ntimes),'-.g',1:Ntimes,repmat(mean(acc11_AZP),1,Ntimes),'-.b')
% % % % % % % % % % % % % % % % % xlabel('Iters');
% % % % % % % % % % % % % % % % % ylabel('OA');
% % % % % % % % % % % % % % % % % set(gca,'xtick',0:2:Ntimes);
% % % % % % % % % % % % % % % % % set(gca,'ytick',0:0.01:1);
% % % % % % % % % % % % % % % % % xtickangle(-45)
% % % % % % % % % % % % % % % % % % % % % %%
% % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % save order.mat
