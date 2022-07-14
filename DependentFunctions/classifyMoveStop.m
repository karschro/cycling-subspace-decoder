function decMove = classifyMoveStop(pMove,thresh)

decMove = zeros(1,length(pMove));
for i = 2:length(pMove)
    if pMove(i) > thresh(2) && decMove(i-1) == 0
        decMove(i) = 1;
    elseif pMove(i) < thresh(1) && decMove(i-1) == 1
        decMove(i) = 0;
    else
        decMove(i) = decMove(i-1);
    end
end