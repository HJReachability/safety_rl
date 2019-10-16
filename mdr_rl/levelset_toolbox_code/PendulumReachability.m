function [ data, g, data0 ] = PendulumReachability(accuracy)
% air3D: demonstrate the 3D aircraft collision avoidance example
%
%   [ data, g, data0 ] = air3D(accuracy)
%
% In this example, the target set is a circle at the origin (cylinder in 3D)
% that represents a collision in relative coordinates between the evader
% (player a, fixed at the origin facing right) and the pursuer (player b).
%
% The relative coordinate dynamics are
%
%   \dot x    = -v_a + v_b \cos \psi + a y
%	  \dot y    = v_b \sin \psi - a x
%	  \dot \psi = b - a
%
% where v_a and v_b are constants, input a is trying to avoid the target
%	input b is trying to hit the target.
%
% For more details, see my PhD thesis, section 3.1.
%
% This function was originally designed as a script file, so most of the
% options can only be modified in the file.  For example, edit the file to
% change the grid dimension, boundary conditions, aircraft parameters, etc.
%
% To get exactly the result from the thesis choose:
%   targetRadius = 5, velocityA = velocityB = 5, inputA = inputB = +1.
%
% Input Parameters:
%
%   accuracy: Controls the order of approximations.
%     'low': Use odeCFL1 and upwindFirstFirst.
%     'medium': Use odeCFL2 and upwindFirstENO2 (default).
%     'high': Use odeCFL3 and upwindFirstENO3.
%     'veryHigh': Use odeCFL3 and upwindFirstWENO5.
%
% Output Parameters:
%
%   data: Implicit surface function at t_max.
%
%   g: Grid structure on which data was computed.
%
%   data0: Implicit surface function at t_0.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing
%   agreement contained in the file LICENSE in the top directory of
%   the distribution.
%
% Ian Mitchell, 3/26/04
% Subversion tags for version control purposes.
% $Date: 2012-07-04 14:27:00 -0700 (Wed, 04 Jul 2012) $
% $Id: air3D.m 74 2012-07-04 21:27:00Z mitchell $

%---------------------------------------------------------------------------
% You will see many executable lines that are commented out.
%   These are included to show some of the options available; modify
%   the commenting to modify the behavior.

%---------------------------------------------------------------------------
% Make sure we can see the kernel m-files.
%run('../addPathToKernel');

%---------------------------------------------------------------------------
% Integration parameters.
tMax = 2;                  % End time.
plotSteps = 9;               % How many intermediate plots to produce?
t0 = 0;                      % Start time.
singleStep = 0;              % Plot at each timestep (overrides tPlot).

% Period at which intermediate plots should be produced.
tPlot = (tMax - t0) / (plotSteps - 1);

% How close (relative) do we need to get to tMax to be considered finished?
small = 100 * eps;

% What kind of dissipation?
dissType = 'global';

%---------------------------------------------------------------------------
% Problem Parameters.
%   targetRadius  Radius of target circle (positive).
%   velocityA	  Speed of the evader (positive constant).
%   velocityB	  Speed of the pursuer (positive constant).
%   inputA	  Maximum turn rate of the evader (positive).
%   inputB	  Maximum turn rate of the pursuer (positive).

 %targetRadius = 5;
% velocityA = 5;
% velocityB = 5;
inputA = 25;
inputB = 5;

choice=[1 2 ];
fix=[3 4];
% choice=[2 4 ];
% fix=[1 3];

%---------------------------------------------------------------------------
% What level set should we view?
level = 0;

% Visualize the 3D reachable set.
displayType = 'contour';

% Pause after each plot?
pauseAfterPlot = 0;

% Delete previous plot before showing next?
deleteLastPlot = 1;

% Visualize the angular dimension a little bigger.
%aspectRatio = [ 1 1 .4 ];

% Plot in separate subplots (set deleteLastPlot = 0 in this case)?
useSubplots = 0;


gravity = 9.8;
masscart = 1.0;
masspole = 0.1;
total_mass = (masspole + masscart);
semi_length = 0.5; % actually half the pole's length
polemass_length = (masspole * semi_length);
force_mag = 10.0;
tau = 0.02;

schemeData.gravity = gravity;
schemeData.masscart = masscart;
schemeData.masspole = masspole;
schemeData.total_mass = total_mass;
schemeData.semi_length = semi_length;
schemeData.polemass_length = polemass_length;
schemeData.force_mag = force_mag;
schemeData.tau = tau;

%---------------------------------------------------------------------------
% Approximately how many grid cells?
%   (Slightly different grid cell counts will be chosen for each dimension.)
Nx = 22;
x_threshold = 2.4;
theta_threshold = 12*2*pi/360;

% Create the grid.
g.dim = 4;
g.min = [  -2.5; -10;  -1.2*pi/2; -5]; %CHANGED
g.max = [ +2.5; +10; +1.2*pi/2; +5]; %CHANGED
g.bdry = { @addGhostExtrapolate; @addGhostExtrapolate; @addGhostExtrapolate; @addGhostExtrapolate };
% Roughly equal dx in x and y (so different N).
g.N = [ Nx; Nx; Nx; Nx];
% Need to trim max bound in \psi (since the BC are periodic in this dimension).
%g.max(3) = g.max(3) * (1 - 1 / g.N(3));
g = processGrid(g);

%---------------------------------------------------------------------------
target_min = [-x_threshold; -Inf; -theta_threshold; -Inf]
target_max = [+x_threshold; +Inf; +theta_threshold; +Inf]
data = -shapeRectangleByCorners(g, target_min, target_max); % target avoid set use negative shapeRectangle so that stays inside rectangle - NL
data0 = data;

%---------------------------------------------------------------------------
% Set up spatial approximation scheme.
schemeFunc = @termLaxFriedrichs;
schemeData.hamFunc = @air3DHamFunc;
schemeData.partialFunc = @air3DPartialFunc;
schemeData.grid = g;

% The Hamiltonian and partial functions need problem parameters.
%schemeData.velocityA = velocityA;
%schemeData.velocityB = velocityB;
schemeData.inputA = inputA;
schemeData.inputB = inputB;

%---------------------------------------------------------------------------
% Choose degree of dissipation.

switch(dissType)
 case 'global'
  schemeData.dissFunc = @artificialDissipationGLF;
 case 'local'
  schemeData.dissFunc = @artificialDissipationLLF;
 case 'locallocal'
  schemeData.dissFunc = @artificialDissipationLLLF;
 otherwise
  error('Unknown dissipation function %s', dissFunc);
end

%---------------------------------------------------------------------------
if(nargin < 1)
  accuracy = 'medium';
end

% Set up time approximation scheme.
integratorOptions = odeCFLset('factorCFL', 0.75, 'stats', 'on');

% Choose approximations at appropriate level of accuracy.
switch(accuracy)
 case 'low'
  schemeData.derivFunc = @upwindFirstFirst;
  integratorFunc = @odeCFL1;
 case 'medium'
  schemeData.derivFunc = @upwindFirstENO2;
  integratorFunc = @odeCFL2;
 case 'high'
  schemeData.derivFunc = @upwindFirstENO3;
  integratorFunc = @odeCFL3;
 case 'veryHigh'
  schemeData.derivFunc = @upwindFirstWENO5;
  integratorFunc = @odeCFL3;
 otherwise
  error('Unknown accuracy level %s', accuracy);
end

if(singleStep)
  integratorOptions = odeCFLset(integratorOptions, 'singleStep', 'on');
end

%---------------------------------------------------------------------------
% Restrict the Hamiltonian so that reachable set only grows.
%   The Lax-Friedrichs approximation scheme MUST already be completely set up.
innerFunc = schemeFunc;
innerData = schemeData;
clear schemeFunc schemeData;

% Wrap the true Hamiltonian inside the term approximation restriction routine.
schemeFunc = @termRestrictUpdate;
schemeData.innerFunc = innerFunc;
schemeData.innerData = innerData;
schemeData.positive = 0;

%---------------------------------------------------------------------------
% Initialize Display

% Set up subplot parameters if necessary.
if(useSubplots)
  rows = ceil(sqrt(plotSteps));
  cols = ceil(plotSteps / rows);
  plotNum = 1;
  subplot(rows, cols, plotNum);
end

[flat_data, flat]=CollapseDown(data,g,choice,fix);

h = visualizeLevelSet(flat, flat_data, displayType, level, [ 't = ' num2str(t0) ]);

camlight right;  camlight left;
hold on;
axis(g.axis);
%daspect(aspectRatio);
drawnow;

%---------------------------------------------------------------------------
% Loop until tMax (subject to a little roundoff).
tNow = t0;
startTime = cputime;

while(tMax - tNow > small * tMax)

  % Reshape data array into column vector for ode solver call.
  y0 = data(:);

  % How far to step?
  tSpan = [ tNow, min(tMax, tNow + tPlot) ];

  % Take a timestep.
  [ t y ] = feval(integratorFunc, schemeFunc, tSpan, y0,...
                  integratorOptions, schemeData);
  tNow = t(end);

  % Get back the correctly shaped data array
  data = reshape(y, g.shape);

  if(pauseAfterPlot)
    % Wait for last plot to be digested.
    pause;
  end

  % Get correct figure, and remember its current view.
  [ view_az, view_el ] = view;

  % Delete last visualization if necessary.
  if(deleteLastPlot)
    delete(h);
  end

  % Move to next subplot if necessary.
  if(useSubplots)
    plotNum = plotNum + 1;
    subplot(rows, cols, plotNum);
  end

  % Create new visualization.

    [flat_data, flat]=CollapseDown(data,g,choice,fix);

    h = visualizeLevelSet(flat, flat_data, displayType, level, [ 't = ' num2str(tNow) ]);


    %load pen_stuff_nominal

    %old=pen_reach_set;

    pen_reach_set=data;
    %look=pen_reach_set(pen_reach_set>0)-old(pen_reach_set>0);
    %max(abs(look(:)))
    %max(pen_reach_set(:))

    pen_grid=g;
    pen_costates=extractCostates(g,data,innerData.derivFunc); %CHANGED

    maxU= inputA;
    save pen_stuff_nominal pen_costates pen_grid pen_reach_set maxU


  % Restore view.
  view(view_az, view_el);

  %CHANGED: If no significant change, call it converged and quit.
  change_amount = max(abs(y - y0));
  fprintf('	Max change in value function %g\n', change_amount)
  if change_amount < 0.1,
      fprintf('	Converged: Threshold reached.\n')
      break;
  end

end

endTime = cputime;
fprintf('Total execution time %g seconds\n', endTime - startTime);

%pen_reach_set=data;
% look=pen_reach_set(pen_reach_set>0)-old(pen_reach_set>0);
% max(abs(look(:)))

%max(pen_reach_set(:))

%pen_grid=g;
%pen_costates=extractCostates(g,data);

%maxU= inputA;
%save pen_stuff_nominal pen_costates pen_grid pen_reach_set maxU


%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function hamValue = air3DHamFunc(t, data, deriv, schemeData)
% air3DHamFunc: analytic Hamiltonian for 3D collision avoidance example.
%
% hamValue = air3DHamFunc(t, data, deriv, schemeData)
%
% This function implements the hamFunc prototype for the three dimensional
%   aircraft collision avoidance example (also called the game of
%   two identical vehicles).
%
% It calculates the analytic Hamiltonian for such a flow field.
%
% Parameters:
%   t            Time at beginning of timestep (ignored).
%   data         Data array.
%   deriv	 Cell vector of the costate (\grad \phi).
%   schemeData	 A structure (see below).
%
%   hamValue	 The analytic hamiltonian.
%
% schemeData is a structure containing data specific to this Hamiltonian
%   For this function it contains the field(s):
%
%   .grid	 Grid structure.
%   .velocityA	 Speed of the evader (positive constant).
%   .velocityB	 Speed of the pursuer (positive constant).
%   .inputA	 Maximum turn rate of the evader (positive).
%   .inputB	 Maximum turn rate of the pursuer (positive).
%
% Ian Mitchell 3/26/04


%some initial parameters for pendulum model



%checkStructureFields(schemeData, 'grid', 'velocityA', 'velocityB', ...
                   %              'inputA', 'inputB');



% implements equation (3.3) from my thesis term by term
%   with allowances for \script A and \script B \neq [ -1, +1 ]
%   where deriv{i} is p_i
%         x_r is grid.xs{1}, y_r is grid.xs{2}, \psi_r is grid.xs{3}
%         v_a is velocityA, v_b is velocityB,
%         \script A is inputA and \script B is inputB
% hamValue = -(-schemeData.velocityA * deriv{1} ...
% 	     + schemeData.velocityB * cos(grid.xs{3}) .* deriv{1} ...
% 	     + schemeData.velocityB * sin(grid.xs{3}) .* deriv{2} ...
% 	     + schemeData.inputA * abs(grid.xs{2} .* deriv{1} ...
%                                     - grid.xs{1} .* deriv{2} - deriv{3})...
% 	     - schemeData.inputB * abs(deriv{3}));



% temp=(-deriv{2}./(M_pen-(mp_pen.*l_pen.*cos(grid.xs{3}).^2)/L_pen)...
%          +cos(grid.xs{3})./M_pen./(L_pen-mp_pen.*l_pen.*cos(grid.xs{3}).^2./M_pen).*deriv{4});
%
% hamValue = -(grid.xs{2} .* deriv{1} ...
% 	     + ((-mp_pen.*l_pen.*g_pen/L_pen.*cos(grid.xs{3}).*sin(grid.xs{3})+mp_pen.*l_pen.*sin(grid.xs{3}).*grid.xs{4}.^2)...
%          ./(M_pen-(mp_pen.*l_pen.*cos(grid.xs{3}).^2)/L_pen)).* deriv{2} ...
% 	     + grid.xs{4}.* deriv{3} ...
%          +(g_pen.*sin(grid.xs{3})-mp_pen.*l_pen.*grid.xs{4}.^2/M_pen.*cos(grid.xs{3}).*sin(grid.xs{3}))...
%          ./(L_pen-mp_pen.*l_pen.*cos(grid.xs{3}).^2./M_pen).*deriv{4}...
% 	     + abs(deriv{2}./(M_pen-(mp_pen.*l_pen.*cos(grid.xs{3}).^2)/L_pen)...
%          -cos(grid.xs{3})./M_pen./(L_pen-mp_pen.*l_pen.*cos(grid.xs{3}).^2./M_pen).*deriv{4}).*schemeData.inputA ...
%          -abs(temp).*(sign(temp)==-sign(grid.xs{2})).*schemeData.inputB);
%         %-abs(temp).*schemeData.inputB);
grid = schemeData.grid;
temp_low = (-schemeData.force_mag + schemeData.polemass_length .* grid.xs{4} .* grid.xs{4} .* sin(grid.xs{3})) ./ schemeData.total_mass;
temp_high = (schemeData.force_mag + schemeData.polemass_length .* grid.xs{4} .* grid.xs{4} .* sin(grid.xs{3})) ./ schemeData.total_mass;
theta_acc_low = (schemeData.gravity .* sin(grid.xs{3}) - cos(grid.xs{3}) .* temp_low) ./ (schemeData.semi_length .* (4.0./3.0 - schemeData.masspole .* cos(grid.xs{3}) .* cos(grid.xs{3}) ./ schemeData.total_mass));
theta_acc_high = (schemeData.gravity .* sin(grid.xs{3}) - cos(grid.xs{3}) .* temp_high) ./ (schemeData.semi_length .* (4.0./3.0 - schemeData.masspole .* cos(grid.xs{3}) .* cos(grid.xs{3}) ./ schemeData.total_mass));
pos_acc_low = temp_low - schemeData.polemass_length .* theta_acc_low .* cos(grid.xs{3}) ./ schemeData.total_mass;
pos_acc_high = temp_high - schemeData.polemass_length .* theta_acc_high .* cos(grid.xs{3}) ./ schemeData.total_mass;
hamValue = (deriv{1} .* grid.xs{2} + deriv{3} .* grid.xs{4} + max(deriv{2} .* pos_acc_high + deriv{4} .* theta_acc_high, deriv{2} .* pos_acc_low + deriv{4} .* theta_acc_low));

%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function alpha = air3DPartialFunc(t, data, derivMin, derivMax, schemeData, dim)
% air3DPartialFunc: Hamiltonian partial fcn for 3D collision avoidance example.
%
% alpha = air3DPartialFunc(t, data, derivMin, derivMax, schemeData, dim)
%
% This function implements the partialFunc prototype for the three dimensional
%   aircraft collision avoidance example (also called the game of
%   two identical vehicles).
%
% It calculates the extrema of the absolute value of the partials of the
%   analytic Hamiltonian with respect to the costate (gradient).
%
% Parameters:
%   t            Time at beginning of timestep (ignored).
%   data         Data array.
%   derivMin	 Cell vector of minimum values of the costate (\grad \phi).
%   derivMax	 Cell vector of maximum values of the costate (\grad \phi).
%   schemeData	 A structure (see below).
%   dim          Dimension in which the partial derivatives is taken.
%
%   alpha	 Maximum absolute value of the partial of the Hamiltonian
%		   with respect to the costate in dimension dim for the
%                  specified range of costate values (O&F equation 5.12).
%		   Note that alpha can (and should) be evaluated separately
%		   at each node of the grid.
%
% schemeData is a structure containing data specific to this Hamiltonian
%   For this function it contains the field(s):
%
%   .grid	 Grid structure.
%   .velocityA	 Speed of the evader (positive constant).
%   .velocityB	 Speed of the pursuer (positive constant).
%   .inputA	 Maximum turn rate of the evader (positive).
%   .inputB	 Maximum turn rate of the pursuer (positive).
%
% Ian Mitchell 3/26/04

%checkStructureFields(schemeData, 'grid', 'velocityA', 'velocityB', ...
                                 %'inputA', 'inputB');

grid = schemeData.grid;
temp_low = (-schemeData.force_mag + schemeData.polemass_length .* grid.xs{4} .* grid.xs{4} .* sin(grid.xs{3})) ./ schemeData.total_mass;
temp_high = (schemeData.force_mag + schemeData.polemass_length .* grid.xs{4} .* grid.xs{4} .* sin(grid.xs{3})) ./ schemeData.total_mass;
theta_acc_low = (schemeData.gravity .* sin(grid.xs{3}) - cos(grid.xs{3}) .* temp_low) ./ (schemeData.semi_length .* (4.0./3.0 - schemeData.masspole .* cos(grid.xs{3}) .* cos(grid.xs{3}) ./ schemeData.total_mass));
theta_acc_high = (schemeData.gravity .* sin(grid.xs{3}) - cos(grid.xs{3}) .* temp_high) ./ (schemeData.semi_length .* (4.0./3.0 - schemeData.masspole .* cos(grid.xs{3}) .* cos(grid.xs{3}) ./ schemeData.total_mass));
pos_acc_low = temp_low - schemeData.polemass_length .* theta_acc_low .* cos(grid.xs{3}) ./ schemeData.total_mass;
pos_acc_high = temp_high - schemeData.polemass_length .* theta_acc_high .* cos(grid.xs{3}) ./ schemeData.total_mass;
alpha = (abs(grid.xs{2} + grid.xs{4}) + max(abs(pos_acc_high + theta_acc_high), abs(pos_acc_low + theta_acc_low)));

% switch dim
%   case 1
%     alpha = abs(grid.xs{2}) ;
%
%   case 2
%     alpha = abs((-mp_pen.*l_pen.*g_pen/L_pen.*cos(grid.xs{3}).*sin(grid.xs{3})+mp_pen*l_pen.*sin(grid.xs{3}).*grid.xs{4}.^2)...
%          ./(M_pen-(mp_pen.*l_pen.*cos(grid.xs{3}).^2)/L_pen))...
%          +abs(schemeData.inputA ./(M_pen-(mp_pen.*l_pen.*cos(grid.xs{3}).^2)/L_pen)) ...
%          +abs(-schemeData.inputB ./(M_pen-(mp_pen.*l_pen.*cos(grid.xs{3}).^2)/L_pen));
%   case 3
%     alpha =  abs(grid.xs{4});
%
%   case 4
%
%     alpha= abs((g_pen.*sin(grid.xs{3})-mp_pen.*l_pen.*grid.xs{4}.^2/M_pen.*cos(grid.xs{3}).*sin(grid.xs{3}))...
%          ./(L_pen-mp_pen.*l_pen.*cos(grid.xs{3}).^2./M_pen))...
%         +abs(-cos(grid.xs{3})./M_pen./(L_pen-mp_pen.*l_pen.*cos(grid.xs{3}).^2./M_pen)).*schemeData.inputA ...
%         +abs(cos(grid.xs{3})./M_pen./(L_pen-mp_pen.*l_pen.*cos(grid.xs{3}).^2./M_pen)).*schemeData.inputB;
%   otherwise
%     error([ 'Partials for the game of two identical vehicles' ...
%             ' only exist in dimensions 1-3' ]);
%end

%%

% pen_stuff_3
%
% keep out set
%
% abs(position)>2
%
% abs(vel)>20
%
% abs(anglular vel)>20

% pen_stuff_2
%
% keep out set
%
% abs(position)>2.5
%
% abs(vel)>20
%
% abs(anglular vel)>20
% abs(angle)>pi/2