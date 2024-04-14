function [handle handlename] = showPose(channels, skel, axisorder, name)

% SHOWPOSE For drawing a skel representation of 3-D data.
%
%	Description:
%
%	HANDLE = SKELVISUALISE(CHANNELS, SKEL) draws a skeleton
%	representation in a 3-D plot.
%	 Returns:
%	  HANDLE - a vector of handles to the plotted structure.
%	 Arguments:
%	  CHANNELS - the channels to update the skeleton with.
%	  SKEL - the skeleton structure.
%	
%
%	See also
%	SKELMODIFY


%	Copyright (c) 2005, 2006 Neil D. Lawrence
% 	skelVisualise.m CVS version 1.4
% 	skelVisualise.m SVN version 30
% 	last update 2008-01-12T11:32:50.000000Z
%

vals = skel2xyz(skel, channels) ;%* 10
if nargin > 2
  vals2 = vals;
  vals2(:,1) = vals(:,axisorder(1)); vals2(:,2) = vals(:,axisorder(2)); vals2(:,3) = -vals(:,axisorder(3)); % FIXME there is an inverted z axis
  vals = vals2;
end

connect = skelConnectionMatrix(skel);

indices = find(connect);
[I, J] = ind2sub(size(connect), indices);
% handle(1) = plot3(vals(:, 1), vals(:, 2), vals(:, 3), '.');
%axis ij % make sure the left is on the left.
% set(handle(1), 'markersize', 10);
hold on
grid on
for i = 1:length(indices)
  % modify with show part (3d geometrical thing)
  if strncmp(skel.tree(I(i)).name,'L',1)
    c = 'r';
  elseif strncmp(skel.tree(I(i)).name,'R',1)
    c = 'g';
  else 
    c = 'b';
  end
  
  % FIXME 
  handle(i+1) = line([vals(I(i), 1) vals(J(i), 1)], ...
                    [vals(I(i), 2) vals(J(i), 2)], ...
                    [vals(I(i), 3) vals(J(i), 3)],'Color',c);
  set(handle(i+1), 'linewidth', 3);
  
  if  ~strcmp(skel.tree(I(i)).name,'Spine') && ...
      ~strcmp(skel.tree(I(i)).name,'LeftShoulder') && ...
      ~strcmp(skel.tree(I(i)).name,'RightShoulder')
%     text(vals(I(i),1),vals(I(i),2),vals(I(i),3),skel.tree(I(i)).name);
  end
  if exist('name','var') && strcmp(skel.tree(I(i)).name,'Head')
    handlename = text(vals(I(i), 1),vals(I(i), 2),vals(I(i), 3),name);
  end
end
xlabel('x')
ylabel('z')
zlabel('y')
axis on
axis equal
end