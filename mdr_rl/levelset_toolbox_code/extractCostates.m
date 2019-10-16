function p = extractCostates(grid, data, derivFunc)
%function p = calculateCostate(grid, data)
% Estimates the costate p at position x for cost function data on grid g by
% numerically taking partial derivatives along each grid direction.
% Numerical derivatives are taken using the levelset toolbox
% HACK ALERT: for now we assume 4-D system to save coding time

% x1_g=linspace(g.min(1),g.max(1),g.N(1));
% x2_g=linspace(g.min(2),g.max(2),g.N(2));
% x3_g=linspace(g.min(3),g.max(3),g.N(3));
% x4_g=linspace(g.min(4),g.max(4),g.N(4));
%
% [derivL, derivR] = upwindFirstWENO5(grid, data, 1);
% grad_levelset1 = (derivL + derivR)/2;
% [derivL, derivR] = upwindFirstWENO5(grid, data, 2);
% grad_levelset2 = (derivL + derivR)/2;
% [derivL, derivR] = upwindFirstWENO5(grid, data, 3);
% grad_levelset3 = (derivL + derivR)/2;
% [derivL, derivR] = upwindFirstWENO5(grid, data, 4);
% grad_levelset4 = (derivL + derivR)/2;
% [derivL, derivR] = upwindFirstWENO5(grid, data, 5);
% grad_levelset5 = (derivL + derivR)/2;



% p{1} = grad_levelset1;
% p{2} = grad_levelset2;
% p{3} = grad_levelset3;
% p{4} = grad_levelset4;
%p{5} = grad_levelset5;

for k=1:grid.dim
    %[derivL, derivR] = upwindFirstWENO5(grid, data, k);
    [derivL, derivR] = feval(derivFunc, grid, data, k); %CHANGED
    p{k}= (derivL + derivR)/2;
end

% p1 = interp3(x2_g,x1_g,x3_g,grad_levelset1,x2r,x1r,x3r);
% p2 = interp3(x2_g,x1_g,x3_g,grad_levelset2,x2r,x1r,x3r);
% p3 = interp3(x2_g,x1_g,x3_g,grad_levelset3,x2r,x1r,x3r);
%
%
% k = length(x); % number of directions we need to take partials
% p = zeros(k, 1);
%
% x1_g=linspace(g.min(1),g.max(1),g.N(1));
% x2_g=linspace(g.min(2),g.max(2),g.N(2));
% x3_g=linspace(g.min(3),g.max(3),g.N(3));
% x4_g=linspace(g.min(4),g.max(4),g.N(4));
%
% % calculate the partial in the x1 direction
% dx = mean(diff(x1_g));
% Vju = interpn(x1_g,x2_g,x3_g,x4_g,data,x(1)+dx, x(2), x(3), x(4));
% Vjl = interpn(x1_g,x2_g,x3_g,x4_g,data,x(1)-dx, x(2), x(3), x(4));
% p(1) = (Vju - Vjl) / (2 * dx);
%
% % calculate the partial in the x1 direction
% dx = mean(diff(x2_g));
% Vju = interpn(x1_g,x2_g,x3_g,x4_g,data,x(1), x(2)+dx, x(3), x(4));
% Vjl = interpn(x1_g,x2_g,x3_g,x4_g,data,x(1), x(2)-dx, x(3), x(4));
% p(2) = (Vju - Vjl) / (2 * dx);
%
% % calculate the partial in the x1 direction
% dx = mean(diff(x3_g));
% Vju = interpn(x1_g,x2_g,x3_g,x4_g,data,x(1), x(2), x(3)+dx, x(4));
% Vjl = interpn(x1_g,x2_g,x3_g,x4_g,data,x(1), x(2), x(3)-dx, x(4));
% p(3) = (Vju - Vjl) / (2 * dx);
%
% % calculate the partial in the x1 direction
% dx = mean(diff(x4_g));
% Vju = interpn(x1_g,x2_g,x3_g,x4_g,data,x(1), x(2), x(3), x(4)+dx);
% Vjl = interpn(x1_g,x2_g,x3_g,x4_g,data,x(1), x(2), x(3), x(4)-dx);
% p(4) = (Vju - Vjl) / (2 * dx);