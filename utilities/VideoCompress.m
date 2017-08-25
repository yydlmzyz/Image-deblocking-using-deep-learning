%该程序参考 zongzong1113 的程序，做了小的修改，
% MPEG压缩编码算法包括了帧内编码、帧间编码，DCT变换编码、自适应量化、熵编码和运动估计和运动补偿
% 等一系列压缩算法,但是该程序没有涉及运动估计、补偿与Huffman熵编码
%该程序在前一篇JPEG压缩与解压程序的基础上可以很快得到。
clear all;
close all;
clc;

filename = 'C:\Users\YY\Desktop\Data\Video2.avi';
%data_rar= 'C:\Users\YY\Desktop\Data\datarar';
T = dctmtx(8);
lighttable=...
    [16 11 10 16 24 40 51 61 ; 
    12 12 14 19 26 58 60 55 ; 
    14 13 16 24 40 57 69 56 ; 
    14 17 22 29 51 87 80 62 ; 
    18 22 37 56 68 109 103 77;
    24 35 55 64 81 104 113 92; 
    49 64 78 87 103 121 120 101; 
    72 92 95 98 112 100 103 99];
colortable=...
    [17 18 24 47 99 99 99 99 ; 
    18 21 26 66 99 99 99 99 ; 
    24 26 56 99 99 99 99 99 ; 
    47 66 99 99 99 99 99 99 ; 
    99 99 99 99 99 99 99 99 ; 
    99 99 99 99 99 99 99 99 ; 
    99 99 99 99 99 99 99 99 ; 
    99 99 99 99 99 99 99 99];
sequence=[1 9 2 3 10 17 25 18 11 4 5 12 19 26 33 41 34 27 20 13 6 7 ...
    14 21 28 35 42 49 57 50 43 36 29 22 15 8 16 23 30 37 44 51 58 59 ...
    52 45 38 31 24 32 39 46 53 60 61 54 47 40 48 55 62 63 56 64];

fprintf('正在读取视频... \n');
video = VideoReader(filename);%读取视频信息
fs = video.FrameRate;%每秒帧数,帧速率 9.0738 frame/s
frame = video.NumberOfFrames;      %帧数
row = video.Height;
col = video.Width;
% Preallocate movie structure.
finput(1:frame) = ...
    struct('cdata', zeros(row, col, 3, 'uint8'),...
           'colormap', []);
% Read one frame at a time.
for k = 1 : frame
    finput(k).cdata = read(video, k);
end

fprintf('读取视频完成... \n');

dim = 3;%彩色视频
%8*8块的数量
r = ceil(row/8);
c = ceil(col/8);

%参考帧，暂时未用，运动估计
imref = zeros(row,col,dim);
%前后帧图片差分编码
fprintf('开始处理视频... \n');


mpeg_Y_rar = int8(zeros(r*c,64,frame));
mpeg_Cb_rar = int8(zeros(r*c,64,frame));
mpeg_Cr_rar = int8(zeros(r*c,64,frame));
max_Y_len = 0;   %量化后所有帧在该列后全为0，故可以截取
max_Cb_len = 0;
max_Cr_len = 0;
temp1 = mod(size(imref,1),8);
temp2 = mod(size(imref,2),8);
for f = 1 : frame
    %pic为前后帧图片差
    pic = uint8((double(finput(f).cdata) - imref + 255)./2);
    pic = rgb2ycbcr(pic);
    %填补图片->行列转化为8的倍数
    if(temp1 ~=0 )
        pic=[pic;uint8(zeros(8-temp1,size(pic,2),3))];
    end
    if(temp2 ~=0 )
        pic = [pic,uint8(zeros(size(pic,1),8-temp2,3))];
    end
    clear temp
    %每一维输入转化为(-128~127)
    t1 = double(pic(:,:,1)) - 2^7;
    t2 = double(pic(:,:,2)) - 2^7;
    t3 = double(pic(:,:,3)) - 2^7;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%DCT变换->量化->z字形编码->DC差分编码%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %处理亮度维
    count = 1;
    p1 = int8(zeros(r*c,64));
    for i = 1 : r
        for j = 1 : c
            temp = round(T*t1(8*i-7:8*i,8*j-7:8*j)*T'./lighttable);%DCT变换,量化   
            p1(count,:) = temp(sequence);%z字形编码
            count = count+1;
        end
    end
    p1(:,1)=[p1(1);diff(p1(:,1))];%DC系数差分编码
    clear t1;
    
    tmp = [];
    for i = 1 : 64   %matlab中 a != b是不对的，应该 a ~= b
    %if  length( find( p1(:,i) ) ~= 0 ) ~= 0    %如果p1第i列全为0，则find()返回空矩阵[]
        if  length( find( p1(:,i) ) ) ~= 0
            tmp = [tmp,i];                           
        end
    end
    p1 = p1(:,tmp);
    col1 = uint8(tmp); 

    if size(p1,2) > max_Y_len
      max_Y_len =  size(p1,2);
    end
   
   mpeg_Y_rar(:,1:size(p1,2),f) = p1;
   col_Y_flag(f,1:size(col1,2)) = col1;
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
   %处理色度维Cb
    count = 1;
    p1 = int8(zeros(r*c,64));
    for i = 1 : r
        for j = 1 : c
            temp = round(T*t2(8*i-7:8*i,8*j-7:8*j)*T'./colortable);
            p1(count,:) = temp(sequence);
            count = count + 1;
        end
    end
    p1(:,1)=[p1(1);diff(p1(:,1))];%DC系数差分编码
    clear t2;  
    
    tmp = [];
    for i = 1 : 64   
        if  length( find( p1(:,i) ) ) ~= 0
            tmp = [tmp,i];                           
        end
    end
    p1 = p1(:,tmp);
    col1 = uint8(tmp); 

    if size(p1,2) > max_Cb_len
      max_Cb_len =  size(p1,2);
    end

    mpeg_Cb_rar(:,1:size(p1,2),f) = p1;
    col_Cb_flag(f,1:size(col1,2)) = col1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    %处理色度维Cr
    count = 1;
    p1 = int8(zeros(r*c,64));
    for i = 1 : r
        for j = 1 : c
            temp = round(T*t3(8*i-7:8*i,8*j-7:8*j)*T'./colortable);
            p1(count,:) = temp(sequence);
            count = count + 1;
        end
    end
    p1(:,1) = [p1(1);diff(p1(:,1))];%DC系数差分编码
    clear t2;  
    
    tmp = [];
    for i = 1 : 64  
        if  length( find( p1(:,i) ) ) ~= 0
            tmp = [tmp,i];                           
        end
    end
    p1 = p1(:,tmp);
    col1 = uint8(tmp); 

    if size(p1,2) > max_Cr_len
      max_Cr_len =  size(p1,2);
    end

    mpeg_Cr_rar(:,1:size(p1,2),f) = p1;
    col_Cr_flag(f,1:size(col1,2)) = col1;
end

mpeg_Y_rar = mpeg_Y_rar(:,1:max_Y_len,:);
col_Y_flag = col_Y_flag(:,1:max_Y_len);
mpeg_Cb_rar = mpeg_Cb_rar(:,1:max_Cb_len,:);
col_Cb_flag = col_Cb_flag(:,1:max_Cb_len);
mpeg_Cr_rar = mpeg_Cr_rar(:,1:max_Cr_len,:);
col_Cr_flag = col_Cr_flag(:,1:max_Cr_len);

fprintf('视频压缩处理结束... \n');

%save data_rar/mpeg_rar mpeg_Y_rar mpeg_Cb_rar mpeg_Cr_rar
%save data_rar/col_flag col_Y_flag col_Cb_flag col_Cr_flag
%save data_rar/r_c r c fs 


save mpeg_rar mpeg_Y_rar mpeg_Cb_rar mpeg_Cr_rar
save col_flag col_Y_flag col_Cb_flag col_Cr_flag
save r_c r c fs 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  解压缩
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('开始视频解压... \n');
clear;
T = dctmtx(8);
lighttable=...
    [16 11 10 16 24 40 51 61 ; 
    12 12 14 19 26 58 60 55 ; 
    14 13 16 24 40 57 69 56 ; 
    14 17 22 29 51 87 80 62 ; 
    18 22 37 56 68 109 103 77;
    24 35 55 64 81 104 113 92; 
    49 64 78 87 103 121 120 101; 
    72 92 95 98 112 100 103 99];
colortable=...
    [17 18 24 47 99 99 99 99 ; 
    18 21 26 66 99 99 99 99 ; 
    24 26 56 99 99 99 99 99 ; 
    47 66 99 99 99 99 99 99 ; 
    99 99 99 99 99 99 99 99 ; 
    99 99 99 99 99 99 99 99 ; 
    99 99 99 99 99 99 99 99 ; 
    99 99 99 99 99 99 99 99];
sequence2=[1 3 4 10 11 21 22 36 2 5 9 12 20 23 35 37 6 8 13 19 24 34 ...
    38 49 7 14 18 25 33 39 48 50 15 17 26 32 40 47 51 58 16 27 31 41 ...
    46 52 57 59 28 30 42 45 53 56 60 63 29 43 44 54 55 61 62 64];

%load data_rar/mpeg_rar; 
%load data_rar/col_flag;
%load data_rar/r_c;

load mpeg_rar; 
load col_flag;
load r_c;


frame = size(col_Y_flag,1);%帧数

for f = 1 : frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%还原亮度Y
    p1 = mpeg_Y_rar(:,:,f); %第f帧
    for i = 2 : size(p1,1)               %直流分量
        p1(i,1) = p1(i-1,1) + p1(i,1);  
    end
    pp1 = int8(zeros(size(p1,1),64));%int8    -128 -- 127
    len = length( find(col_Y_flag(f,:)) );
    pp1(:,col_Y_flag(f,1:len)) = p1(:,1:len);
    p1 = pp1;
    
    count = 1;
    t1 = zeros(8*r,8*c);
    for i = 1 : r
        for j = 1 : c
            tmp = p1(count,:);
            tmp = reshape(tmp(sequence2),[8,8]);%z字形解码，确实是这样的
            t1(8*i-7:8*i,8*j-7:8*j) = T'*(double(tmp).*lighttable)*T;%反量化,DCT逆变换
            count = count + 1;
        end
    end 
    mov(:,:,1,f) = uint8(t1 + 2^7); %还原一帧图像
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%还原色度Cb
    p1 = mpeg_Cb_rar(:,:,f); %第f帧
    for i = 2 : size(p1,1)               %直流分量
        p1(i,1) = p1(i-1,1) + p1(i,1);  
    end
    pp1 = int8(zeros(size(p1,1),64));%int8    -128 -- 127
    len = length( find(col_Cb_flag(f,:)) );
    pp1(:,col_Cb_flag(f,1:len)) = p1(:,1:len);
    p1 = pp1;
    
    count = 1;
    t1 = zeros(8*r,8*c);
    for i = 1 : r
        for j = 1 : c
            tmp = p1(count,:);
            tmp = reshape(tmp(sequence2),[8,8]);%z字形解码，确实是这样的
            t1(8*i-7:8*i,8*j-7:8*j) = T'*(double(tmp).*colortable)*T;%反量化,DCT逆变换
            count = count + 1;
        end
    end 
    mov(:,:,2,f) = uint8(t1 + 2^7); %还原一帧图像   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%还原色度Cr
    p1 = mpeg_Cr_rar(:,:,f); %第f帧
    for i = 2 : size(p1,1)               %直流分量
        p1(i,1) = p1(i-1,1) + p1(i,1);  
    end
    pp1 = int8(zeros(size(p1,1),64));%int8    -128 -- 127
    len = length( find(col_Cr_flag(f,:)) );
    pp1(:,col_Cr_flag(f,1:len)) = p1(:,1:len);
    p1 = pp1;
    
    count = 1;
    t1 = zeros(8*r,8*c);
    for i = 1 : r
        for j = 1 : c
            tmp = p1(count,:);
            tmp = reshape(tmp(sequence2),[8,8]);%z字形解码，确实是这样的
            t1(8*i-7:8*i,8*j-7:8*j) = T'*(double(tmp).*colortable)*T;%反量化,DCT逆变换
            count = count + 1;
        end
    end 
    mov(:,:,3,f) = uint8(t1 + 2^7); %还原一帧图像     
    mov(:,:,:,f) = ycbcr2rgb(mov(:,:,:,f));%转到RGB空间
end
%pic = ycbcr2rgb(mov);
clear mpeg_Y_rar mpeg_Cb_rar mpeg_Cr_rar;
fprintf('完成视频解压... \n');

%save data_rar/mov mov fs;

save mov mov fs;
clear all;
close all;
clc;

%load data_rar/mov;
load mov;
fprintf('开始播放视频... \n');
frame = size(mov,4);
for f = 1 : frame
    imshow(mov(:,:,:,f));   
    pause(1/fs); 
end
fprintf('播放视频结束... \n');