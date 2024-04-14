path(path, pwd);

if ispc
    disp('Intallation for Windows x64');
    
    path(path, [pwd '/Bin/Win64']);
end

if ismac
    disp('Intallation for Mac Os X x64');
    
    path(path, [pwd '/Bin/Mac64']);
else 
    if isunix
        disp('Intallation for Linux x64');
        
        path(path, [pwd '/Bin/Unix64']);
    end
end


disp('Instalation completed!');