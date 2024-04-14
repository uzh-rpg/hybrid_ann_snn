function updatePose(handles, channels, skel, axisorder, texthandle)
vals = skel2xyz(skel, channels) ;%* 10
if nargin > 3
  vals2 = vals;
  vals2(:,1) = vals(:,axisorder(1)); vals2(:,2) = vals(:,axisorder(2)); vals2(:,3) = -vals(:,axisorder(3));
  vals = vals2;
end

connect = skelConnectionMatrix(skel);

indices = find(connect);
[I, J] = ind2sub(size(connect), indices);

% set(handles(1),'XData',vals(:,1),'YData',vals(:,2),'ZData',vals(:,3));
for i = 1:length(indices)
  set(handles(i+1), 'XData',[vals(I(i), 1) vals(J(i), 1)],'YData',[vals(I(i), 2) vals(J(i), 2)],'ZData',[vals(I(i), 3) vals(J(i), 3)]);
  
  if exist('texthandle','var') && strcmp(skel.tree(I(i)).name,'Head')
    set(texthandle, 'position', [vals(I(i), 1),vals(I(i), 2),vals(I(i), 3)]);
  end
end

end