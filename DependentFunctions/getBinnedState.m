function state = getBinnedState(vel,nBins,Settings)

state = zeros(1,nBins);
for n = 1:nBins
   v = vel((1:Settings.binSize)+(n-1)*Settings.binSize);
   if all(abs(v) > Settings.moveVigorouslyThresh)
       state(n) = 1;
   elseif all(abs(v) < Settings.moveStopThresh)
       state(n) = 0;
   else
       state(n) = 0.5;
   end
end