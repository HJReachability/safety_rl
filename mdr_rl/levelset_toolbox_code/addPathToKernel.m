% Script file that adds the Kernel subdirectories to the path.
%
% Call this script before working with the Matlab Level Set Toolbox
%   (or place it in your startup.m file).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing
%   agreement contained in the file LICENSE in the top directory of
%   the distribution.
%
% Ian Mitchell, 1/13/04

% Proper operation of function handles seems to require an absolute path
%   (at least for Matlab version 6.5).
addpath(genpath('~/ToolboxLS/Kernel'));