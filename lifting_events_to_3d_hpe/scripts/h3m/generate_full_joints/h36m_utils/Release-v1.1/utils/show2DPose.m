function handle = show2DPose(channels, skel, padding)

% SHOWPOSE For drawing a skel representation of 2-D data.
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
% modified by catalin ionescu


if nargin<3
  padding = 0;
end
channels = [channels zeros(1, padding)];
% modify the skeleton

showtext = false;

vals = bvh2xy(skel, channels) ;%* 10
% vals(:,2) = 1000-vals(:,2);
connect = skelConnectionMatrix(skel);

indices = find(connect);
[I, J] = ind2sub(size(connect), indices);
% handle(1) = plot(vals(:, 1), vals(:, 2), '.');
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
                    [vals(I(i), 2) vals(J(i), 2)],'Color',c);
  set(handle(i+1), 'linewidth', 3);
  
  if  ~strcmp(skel.tree(I(i)).name,'Spine') && ...
      ~strcmp(skel.tree(I(i)).name,'LeftShoulder') && ...
      ~strcmp(skel.tree(I(i)).name,'RightShoulder') && showtext
%     text(vals(I(i),1),vals(I(i),2),skel.tree(I(i)).name);
  end
end
xlabel('x')
ylabel('z')
axis on
axis equal
end