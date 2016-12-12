clear all;
close all;

run('/home/lci/vlfeat-0.9.20/toolbox/vl_setup')


for i=1:261
    rgb_filename = sprintf('/media/lci/storage/KingsCollege/seq1/frame%05d.png', i)

    pfx = fullfile(rgb_filename);
    I = imread(pfx);

   % image(I) ;
    I = single(rgb2gray(I));
    [f,d] = vl_sift(I);
    saved_SIFT_filename= sprintf('/home/lci/PR2017/extractSIFTVLFeat/saved_SIFT/sift_frame%05d.mat',i)
    save(saved_SIFT_filename,'f','d','-v7.3');
   % perm = randperm(size(f,2)) 
   % sel = perm(1:size(f,2)) 
   % h1 = vl_plotframe(f(:,sel)) ;
   % h2 = vl_plotframe(f(:,sel)) ;
   % set(h1,'color','k','linewidth',3) ;
    %set(h2,'color','y','linewidth',2) ;
end
