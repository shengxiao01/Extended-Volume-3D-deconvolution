%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extended Volume 3D (EV-3D) deconvolution algorithm
%
% The following code reproduces the main results of Fig. 2 from paper 
% "High-contrast multifocus microscopy with a single camera and z-splitter
% prism", doi: https://doi.org/10.1101/2020.08.04.236661
%
% by Sheng Xiao, 09/10/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load data and setup parameters
% load data
clear all;close all;
addpath('./code');
load('data.mat');
[XX,YY,ZZ,TT] = size(stack);
% imaging parameters
NA = 0.4; % objective numerical aperture
dz = 20.56; % delta z, um
dx = 1.083; % laterial pixel size, um
lambda = 0.51; % wavelength, um
index = 1.0; % refractive index

% optimization parameters
TV_reg = 0; % amount of TV regularization
xy_pad = 50; % lateral volume extension
z_pad = 8; % axial volume extension
outer_iter = 20; % number of outer iterations of EV-3D
inner_iter = 80; % number of innter iterations of EV-3D

rl_iter = 200; % number of iteratins for traditional RL-3D


%% EV-3D deconvolution
Nx = XX + 2*xy_pad;
Nz = ZZ + 2*z_pad;
psf = ComputeGaussianPSF(NA,lambda,dx,dz,Nx,floor(Nz/2),index);
otf = psf2otf(psf, [Nx,Nx,Nz]);

im_ev = padarray(stack, [xy_pad, xy_pad, z_pad], 'replicate', 'both'); % image over V_EV

im_mask = padarray(stack, [xy_pad, xy_pad, z_pad], 0, 'both');
im_mask = find(im_mask>0); % pixel indices of V_I
obs = stack(stack>0); % corresponding pixel values of V_I

if canUseGPU()
    obs = single(gpuArray(obs));
    im_ev = single(gpuArray(im_ev));
    otf = single(gpuArray(otf));
    im_mask = single(gpuArray(im_mask));
end

tic;
% main EV-3D deconvolution loop
for i = 1:outer_iter
    est = RL_TV(im_ev, otf, inner_iter, TV_reg);
    im_ev = real(ifftn(fftn(est).*otf));
    im_ev = max(im_ev, 0);
    im_ev(im_mask) = obs;
end
toc
EVdeconv = est((xy_pad+1):(xy_pad+XX), (xy_pad+1):(xy_pad+YY),(z_pad+1):(z_pad+ZZ));  % deconvolution results
%% RL-3D deconvolution
psf_rl = ComputeGaussianPSF(NA, lambda, dx, dz, XX, ZZ-1, index);
im_rl = single(padarray(stack, [0, 0, floor(ZZ/2)], 'replicate', 'both'));
otf_rl = single(psf2otf(psf_rl, size(im_rl)));
est_rl = RL_TV(im_rl, otf_rl, rl_iter, TV_reg); % main RL-3D deconvolution
RLdeconv = est_rl(:,:,floor(ZZ/2)+1:floor(ZZ/2) + ZZ);

%% plot results
plotResults(stack, RLdeconv, EVdeconv, depthCode);

