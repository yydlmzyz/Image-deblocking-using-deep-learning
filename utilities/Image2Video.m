temp1='C:\Users\YY\Desktop\Data\BlowingBubble\'% define the folder of images
temp2='.png';
WriterObj=VideoWriter('C:\Users\YY\Desktop\Data\Video3');% define the video(I write mp4,but the result is avi???,the default is avi,.mp4 no use)
open(WriterObj);
for i=2:500 %the number of images 
frame=imread(strcat(temp1,num2str(i),temp2));
writeVideo(WriterObj,frame);
end
close(WriterObj);

%input 50 images why output video is just 1 second? looks like 30 images per second