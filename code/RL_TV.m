function J = RL_TV(I, otf, maxIter, reg)
% RL deconvolution with TV regularization
% adapted from Matlab deconvlucy function

if ~exist('maxIter', 'var')
    maxIter = 20;
end
if ~exist('reg', 'var')
    reg = 0.01;
end

if canUseGPU()
    if ~isa(I,'gpuArray')
        I = single(gpuArray(I));
    end
    if ~isa(otf,'gpuArray')
        otf = single(gpuArray(otf));
    end
end

sizeI = size(I);

J1 = single(I);
J2 = J1;
J3 = 0;
if canUseGPU()
    J4 = zeros(prod(sizeI), 2, 'single', 'gpuArray');
else
    J4 = zeros(prod(sizeI), 2);
end

wI = max(J1, 0);


lambda = 0;
for k = 1:maxIter
    if k > 2,
        lambda = (dot(J4(:,1),J4(:,2)))/(dot(J4(:,2), J4(:,2)) +eps);
        lambda = max(min(lambda,1),0);% stability enforcement
    end
    Y = max(J2 + lambda*(J2 - J3),0);% plus positivity constraint   

    % 3.b  Make core for the LR estimation
    ReBlurred = real(ifftn(otf.*fftn(Y)));
    ReBlurred = max(ReBlurred, eps);
    %ReBlurred(ReBlurred == 0) = eps;
    ImRatio = wI./ReBlurred + eps;
   
    
    Ratio = real(ifftn(conj(otf).*fftn(ImRatio)));
    if reg ~= 0 % total variation regularization 
        TV_term = computeTV(J2, reg);
        Ratio = Ratio./TV_term; 
    end
    clear ImRatio ReBlurred;

    % 3.c Determine next iteration image & apply positivity constraint
    J3 = J2;
    J2 = max(Y.*Ratio,0);  
    J4 = [J2(:)-Y(:) J4(:,1)];
end

J = gather(J2);

end

function TV_term = computeTV(I, reg)
% calculate TV regularization in the lateral direction for RL deconvlution
% adapted from: https://hal.inria.fr/inria-00070726/document

gx = diff(I, 1, 1); 
Oxp = padarray(gx, [1, 0, 0], 0, 'post');
Oxn = padarray(gx, [1, 0, 0], 0, 'pre');
mx = (sign(Oxp) + sign(Oxn))./2.*min(Oxp, Oxn);
mx = max(mx, eps);
Dx = Oxp./sqrt(Oxp.^2 + mx.^2);
DDx = diff(Dx, 1, 1);
DDx = padarray(DDx, [1, 0, 0], 0, 'pre');

gy = diff(I, 1, 2); 
Oyp = padarray(gy, [0, 1, 0], 0, 'post');
Oyn = padarray(gy, [0, 1, 0], 0, 'pre');
my = (sign(Oyp) + sign(Oyn))./2.*min(Oyp, Oyn);
my = max(my, eps);
Dy = Oyp./sqrt(Oyp.^2 + my.^2);
DDy = diff(Dy, 1, 2);
DDy = padarray(DDy, [0, 1, 0], 0, 'pre');

TV_term = 1 - (DDx + DDy).*reg;

TV_term = max(TV_term, eps);

%disp([max(gmag(:)), max(gmag(:))])
end

