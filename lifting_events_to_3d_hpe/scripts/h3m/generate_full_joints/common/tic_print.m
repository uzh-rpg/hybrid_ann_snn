function tic_print( string, interval )

if nargin < 2
    interval = 1;
end

persistent th;

if isempty(th)
  th = tic();
end

if toc(th) > interval
  fprintf(string);
  drawnow;
  th = tic();
end

end