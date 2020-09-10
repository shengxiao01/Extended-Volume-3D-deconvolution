function plotResults(raw, RL, EV, depthCode)
% calculate color coded image and plot results
figure;

edof = squeeze(sum(raw.*depthCode, 3));
edof = edof - min(edof,[],'all');
edof = edof./max(edof,[],'all');
subplot(1,3,1);imagesc(edof);axis image;colormap(gray);axis off;title('Raw EDOF stack');

edof = squeeze(sum(RL.*depthCode, 3));
edof = edof - min(edof,[],'all');
edof = edof./max(edof,[],'all');
subplot(1,3,2);imagesc(edof);axis image; colormap(gray);axis off;title('RL-3D EDOF stack');


edof = squeeze(sum(EV.*depthCode, 3));
edof = edof - min(edof,[],'all');
edof = edof./max(edof,[],'all');
subplot(1,3,3);imagesc(edof);axis image; colormap(gray);axis off;title('EV-3D EDOF stack')

end