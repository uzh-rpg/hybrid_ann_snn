function showCamera(Camera, axisorder, name)
  if ~exist('axisorder','var')
		axisorder = [1 2 3];
	end
	Length = 1000;
	
	O = Camera.T; 

	VX(1,:) = [Length/3 0 0] * Camera.R;
	VX(2,:) = [0 Length/3 0] * Camera.R;
	VX(3,:) = [0 0 Length] * Camera.R;

	VO(1,:) = O + VX(1,:);
	VO(2,:) = O + VX(2,:);
	VO(3,:) = O + VX(3,:);

	VN(1,:) = VX(1,:) + VX(2,:) + VX(3,:) + O;
	VN(2,:) = VX(1,:) - VX(2,:) + VX(3,:) + O;
	VN(3,:) = -VX(1,:)+ VX(2,:) + VX(3,:) + O;
	VN(4,:) = -VX(1,:)- VX(2,:) + VX(3,:) + O;
	if exist('name','var')
		text(O(axisorder(1)),O(axisorder(2)),O(axisorder(3)),name);
	end
%         plot3([O(axisorder(1)) VO(1,axisorder(1))],[O(axisorder(2)) VO(1,axisorder(2))],[O(axisorder(3)) VO(1,axisorder(3))],'LineWidth',2,'Color','r');
%         plot3([O(axisorder(1)) VO(2,axisorder(1))],[O(axisorder(2)) VO(2,axisorder(2))],[O(axisorder(3)) VO(2,axisorder(3))],'LineWidth',2,'Color','g');
	plot3([O(axisorder(1)) VO(3,axisorder(1))],[O(axisorder(2)) VO(3,axisorder(2))],[O(axisorder(3)) VO(3,axisorder(3))],'LineWidth',2,'Color','r');
	line(VN([1 2 4 3 1],axisorder(1)),VN([1 2 4 3 1],axisorder(2)),VN([1 2 4 3 1],axisorder(3)),'LineWidth',2,'Color',.7*[1 1 1]);
	line([O(axisorder(1)) VN(1,axisorder(1))],[O(axisorder(2)) VN(1,axisorder(2))],[O(axisorder(3)) VN(1,axisorder(3))],'LineWidth',2,'Color',.7*[1 1 1]);
	line([O(axisorder(1)) VN(2,axisorder(1))],[O(axisorder(2)) VN(2,axisorder(2))],[O(axisorder(3)) VN(2,axisorder(3))],'LineWidth',2,'Color',.7*[1 1 1]);
	line([O(axisorder(1)) VN(3,axisorder(1))],[O(axisorder(2)) VN(3,axisorder(2))],[O(axisorder(3)) VN(3,axisorder(3))],'LineWidth',2,'Color',.7*[1 1 1]);
	line([O(axisorder(1)) VN(4,axisorder(1))],[O(axisorder(2)) VN(4,axisorder(2))],[O(axisorder(3)) VN(4,axisorder(3))],'LineWidth',2,'Color',.7*[1 1 1]);      
end