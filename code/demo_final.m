% Paper: Multi-graph Fusion and Learning for RGBT Image Saliency Detection

%% main
imgRoot = './test/';
imgRoot1='.\image\RGB\';%%RGB images input
imgRoot2='.\image\T\';%%Thermal images input
imgRoot3='.\image\RGBT\';%%Thermal images input
saldir='./saliencymap/';% the output path of the saliency map
supdir='./superpixels/';% the superpixel label file path
FCNfeatureRoot1 = '.\FCN\RGB\';%%use pretrained FCN-32S network
FCNfeatureRoot2 = '.\FCN\T\';
FCNfeatureRoot3 = '.\FCN\RGBT\';
boundary1 = '.\boundary\RGB\'; 
boundary2 = '.\boundary\T\';
mkdir(supdir);
mkdir(saldir);
imnames1=dir([imgRoot1 '*' 'jpg']);
imnames2=dir([imgRoot2 '*' 'jpg']);
imnames3=dir([imgRoot3 '*' 'bmp']);
boundaryname1=dir([boundary1 '*' 'png']);
boundaryname2=dir([boundary2 '*' 'png']);
theta1=20;
theta2=40;
theta3=20;
theta4=40;
spnumber=300;
eta=1.8;
%%
for ii=1:length(imnames1)  %%这里假设假设两种假设这模态图片数量是一样的 
    disp(ii);   
    im1=[imgRoot1 imnames1(ii).name];
    im2=[imgRoot2 imnames2(ii).name];     
    img1=imread(im1);
    img2=imread(im2);
    
    im3=[imgRoot3 imnames3(ii).name];
    img3=imread(im3);
     
    b1=[boundary1 boundaryname1(ii).name];
    b2=[boundary2 boundaryname2(ii).name];     
    boun1=imread(b1);
    boun2=imread(b2);
     
    boun1bw=im2bw(boun1,0.3);
    boun2bw=im2bw(boun2,0.3);
  
    num_boun1=find(boun1bw==1);
    num_boun2=find(boun2bw==1);
    [num1,~]=size(num_boun1);
    [num2,~]=size(num_boun2);
    num_img1=(num1/num2)/(num1/num2+1);
    num_img2=1/(num1/num2+1);   
    Simg=num_img1*img1 + num_img2*img2; % generating RGBT image 
    Simgn=[imgRoot imnames1(ii).name];
    Simgname=[Simgn(1:end-4)  '.bmp'];
    imwrite(Simg,Simgname,'bmp');% the slic software support only the '.bmp' image
    [m,n,k]=size(Simg);
   
 %%  
    Simgname=[Simgn(1:end-4)  '.bmp'];
    comm=['SLICSuperpixelSegmentation' ' ' Simgname ' ' int2str(20) ' ' int2str(spnumber) ' ' supdir];
    system(comm);
    spname=[supdir imnames1(ii).name(1:end-4)  '.dat'];
    superpixels=ReadDAT( [m,n],spname);
    spnum=max(superpixels(:));
 %%               
     adjloop=AdjcProcloop(superpixels,spnum);
        edges=[];
        for i=1:spnum
            indext=[];
            ind=find(adjloop(i,:)==1);
            for j=1:length(ind)
                indj=find(adjloop(ind(j),:)==1);
                indext=[indext,indj];
            end
            indext=[indext,ind];
            indext=indext((indext>i));
            indext=unique(indext);
            if(~isempty(indext))
                ed=ones(length(indext),2);
                ed(:,2)=i*ed(:,2);
                ed(:,1)=indext;
                edges=[edges;ed];
            end
        end
        
      inds = cell(spnum,1);
        for ttt=1:spnum
            inds{ttt} = find(superpixels==ttt);
        end
   

    %% RGB
    [G_meanVgg1,G_meanVgg2] = ExtractFCNfeature(FCNfeatureRoot1,imnames1(ii).name(1:end-4),inds,m,n);        
    G_weights1 = makeweights(edges,G_meanVgg1,theta1);    
    G_weights2 = makeweights(edges,G_meanVgg2,theta1);   
    G_W1 = adjacency(edges,G_weights1,spnum); 
    G_W2 = adjacency(edges,G_weights2,spnum);

    %% T 
    [T_meanVgg1,T_meanVgg2] =ExtractFCNfeature(FCNfeatureRoot2,imnames2(ii).name(1:end-4),inds,m,n);
    T_weights1 = makeweights(edges,T_meanVgg1,theta2);    
    T_weights2 = makeweights(edges,T_meanVgg2,theta2);   
    T_W1 = adjacency(edges,T_weights1,spnum); 
    T_W2 = adjacency(edges,T_weights2,spnum);
   
   %% CIE-LAB
    input_vals1 = reshape(img1, m*n, k); 
    input_vals2 = reshape(img2, m*n, k);
    rgb_vals1 = zeros(spnum,1,3);
    rgb_vals2 = zeros(spnum,1,3); 
    for i = 1:spnum
        rgb_vals1(i,1,:) = mean(input_vals1(inds{i},:),1);  
        rgb_vals2(i,1,:) = mean(input_vals2(inds{i},:),1);
    end
    lab_vals1 = colorspace('Lab<-', rgb_vals1);
    lab_vals2 = colorspace('Lab<-', rgb_vals2);
    seg_vals1=reshape(lab_vals1,spnum,3);% lab 颜色特征   
    seg_vals2=reshape(lab_vals2,spnum,3);
    Weights1 = makeweights(edges,seg_vals1,theta3); 
    Weights2 = makeweights(edges,seg_vals2,theta4); 
    W1 = adjacency(edges,Weights1,spnum);  
    W2= adjacency(edges,Weights2,spnum);     
    
   %%  feature RGBT
    input_vals3 = reshape(img3, m*n, k); 
    rgb_vals3 = zeros(spnum,1,3);
    for i = 1:spnum
        rgb_vals3(i,1,:) = mean(input_vals3(inds{i},:),1);    
    end
    lab_vals3 = colorspace('Lab<-', rgb_vals1); 
    seg_vals3=reshape(lab_vals3,spnum,3);% lab 颜色特征 
   
    [RGBT_meanVgg1,RGBT_meanVgg2] = ExtractFCNfeature(FCNfeatureRoot3,imnames3(ii).name(1:end-4),inds,m,n);   
   
    S1=seg_vals3;
    S2=RGBT_meanVgg1;
    S3=RGBT_meanVgg2;
 %%
    A1=zeros(spnum,spnum,3);
    A2=zeros(spnum,spnum,3);  
    A1(:,:,1)=full(G_W1);
    A1(:,:,2)=full(G_W2);
    A1(:,:,3)=full(W1);
    A2(:,:,1)=full(T_W1);
    A2(:,:,2)=full(T_W2);
    A2(:,:,3)=full(W2);
    tau = 0.1;
   
    [Z1,Z2] = LR_Tensor(A1,A2,tau);

    G_W1=Z1(:,:,1);
    G_W2=Z1(:,:,2);
    W1=Z1(:,:,3);
    T_W1=Z2(:,:,1);
    T_W2=Z2(:,:,2);
    W2=Z2(:,:,3);
  
    dd = sum(G_W1); G_D1 = sparse(1:spnum,1:spnum,dd); %clear dd;
    G_L1 =G_D1-G_W1;
    dd = sum(G_W2); G_D2 = sparse(1:spnum,1:spnum,dd);% clear dd;
    G_L2 =G_D2-G_W2;
   
    dd = sum(T_W1); T_D1 = sparse(1:spnum,1:spnum,dd);% clear dd;
    T_L1 =T_D1-T_W1; 
    dd = sum(T_W2); T_D2 = sparse(1:spnum,1:spnum,dd); %clear dd;
    T_L2 =T_D2-T_W2; 
   
    dd1 = sum(W1); D1 = sparse(1:spnum,1:spnum,dd1);%clear dd1;
    dd2 = sum(W2); D2 = sparse(1:spnum,1:spnum,dd2); %clear dd2;
    G_L3=D1-W1;
    T_L3=D2-W2;
  %%
    WeightsS1 = makeweights(edges,S1,20); 
    WS1 = adjacency(edges,WeightsS1,spnum);  
    
    WeightsS2 = makeweights(edges,S2,20); 
    WS2 = adjacency(edges,WeightsS2,spnum); 
    
    WeightsS3 = makeweights(edges,S3,20); 
    WS3 = adjacency(edges,WeightsS3,spnum);  
    
    ddS1 = sum(WS1); DS1 = sparse(1:spnum,1:spnum,ddS1);%clear ddS1;
    LS1=DS1-WS1;
      
    ddS2 = sum(WS2); DS2 = sparse(1:spnum,1:spnum,ddS2);%clear ddS2;
    LS2=DS2-WS2;  
    
    ddS3 = sum(WS3); DS3 = sparse(1:spnum,1:spnum,ddS3);%clear ddS3;
    LS3=DS3-WS3;    
     %% top           
     Yt=zeros(spnum,1);
     bst=unique(superpixels(1,1:n));         
     Yt(bst)=1;
     [St] =FL(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yt,WS1,WS2,WS3);   
     St=(St-min(St(:)))/(max(St(:))-min(St(:)));
     St=1-St;      
    %% down
     Yd=zeros(spnum,1);
     bst=unique(superpixels(m,1:n));         
     Yd(bst)=1;
     [Sd] =FL(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yd,WS1,WS2,WS3);    
     Sd=(Sd-min(Sd(:)))/(max(Sd(:))-min(Sd(:)));
     Sd=1-Sd;
    %% right
     Yr=zeros(spnum,1);
     bst=unique(superpixels(1:m,1));         
     Yr(bst)=1;
     [Sr] =FL(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yr,WS1,WS2,WS3);    
     Sr=(Sr-min(Sr(:)))/(max(Sr(:))-min(Sr(:)));
     Sr=1-Sr; 
    %% left
     Yl=zeros(spnum,1);
     bst=unique(superpixels(1:m,n));         
     Yl(bst)=1;
     [Sl] =FL(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yl,WS1,WS2,WS3); 
     Sl=(Sl-min(Sl(:)))/(max(Sl(:))-min(Sl(:)));
     Sl=1-Sl;
   %% combine 
    Sc=(St.*Sd.*Sl.*Sr); 
    Sc=(Sc-min(Sc(:)))/(max(Sc(:))-min(Sc(:))); 

   %% foreground seeds
     seeds=Sc;     
     th=mean(Sc)*eta;
     seeds(seeds<th)=0;
     seeds(seeds>=th)=1; 
    %% ------------------------stage2----------------------%%   
   [S] =FL2(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,seeds,WS1,WS2,WS3);
   
    mapstage2=zeros(m,n);
    for i=1:spnum
        mapstage2(inds{i})=S(i);
    end
    mapstage2=(mapstage2-min(mapstage2(:)))/(max(mapstage2(:))-min(mapstage2(:)));
    mapstage2=uint8(mapstage2*255);  
    [Sp] = postprocessing(mapstage2,spnum,inds,boun1,Simg,G_L1,G_L2,G_L3,T_L1,T_L2,T_L3);
    outname2=[saldir imnames1(ii).name(1:end-4) '.jpg'];
    imwrite(Sp , outname2); 
     
     
end

 