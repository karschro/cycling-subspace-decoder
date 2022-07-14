function output = moveOnsetSnippet(fSpikes,moveOnsets,decMove,epoch)

% Check trial for a clean movement onset snippet (and save the last one).
output = [];
for i = 1:length(moveOnsets)
    prepPeriod = moveOnsets(i) + (-200:-1);
    movePeriod = moveOnsets(i) + (1:500);
    snipPeriod = moveOnsets(i) + epoch;
    if min(prepPeriod(1),snipPeriod(1)) < 1 || max(movePeriod(end),snipPeriod(end)) > size(fSpikes,2)
        continue;
    end
    if all(decMove(prepPeriod) == 0) && all(decMove(movePeriod) == 1)
        output = fSpikes(:,snipPeriod);
    end
end