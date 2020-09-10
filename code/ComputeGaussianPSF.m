%%
function psf = ComputeGaussianPSF(NA,lambda,dx,dz,Nx,Nz,index)
% calculate Gaussian-Lorentzian PSF
z = (-Nz:Nz)*dz;  % z distance, in um

k = index/lambda;
delta_k = sqrt(2)*NA*k;

xi = pi.*delta_k.^2.*z./2./k;

L = dx*Nx;
x = (-L/2 + dx/2):dx:(L/2 - dx/2);
[xx,yy] = meshgrid(x, x);
rho = sqrt(xx.^2 + yy.^2);

psf = zeros(Nx,Nx,length(z));
for i = 1:length(z)
psf(:,:,i) = pi.*delta_k^2./(1+xi(i).^2).*exp(-pi.^2.*delta_k.^2.*rho.^2./(1+xi(i).^2));
psf(:,:,i) = psf(:,:,i)./sum(psf(:,:,i),'all');
end
psf = psf./sum(psf,'all');

end
