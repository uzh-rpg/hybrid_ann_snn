function flag = iscluster()
% returns true if the code is run on the cluster and false otherwise

[~, hostname] = system('hostname');
if strcmp(hostname(1:2), 'cn') || strcmp(hostname(1:min(8,length(hostname))),'headnode')
  flag = true;
else
  flag = false;
end
end