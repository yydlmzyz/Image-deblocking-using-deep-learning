%http://blog.csdn.net/anran_zhou/article/details/70241109
video=VideoReader('C:\Users\YY\Desktop\Data\Video2.avi');%从磁盘上载入一段视频（不仅限于.avi形式）
w=video.Width;
h=video.Height;
x=0;
y=0;
for i=1:video.numberofframes%下面对该视频的每一帧进行处理
temp=read(video,i);%获取该视频的第i帧
imshow(temp);%显示第i帧图像
imwrite(temp,strcat('C:\Users\YY\Desktop\Data\',num2str(i),'.jpg'),'jpg');
rectangle('Position',[x,y,w,h],'EdgeColor','r');%顺便带上函数rectangle的用法
pause(0.05);%停5ms相当于OpenCV的cvWaitKey函数
end