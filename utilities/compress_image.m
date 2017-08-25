QP = 10;

home='C:\Users\YY\Desktop\Data\images\';
data_path = dir(fullfile(home,'Train','*.jpg'));
patch_num = length(data_path);
for i=1:patch_num
    im = imread(fullfile(home,'Train',num2str(data_path(i).name)));
    imwrite(im(:,:,:),fullfile(home,'TrainData10',num2str(data_path(i).name)),'jpg','Quality',QP);
end




