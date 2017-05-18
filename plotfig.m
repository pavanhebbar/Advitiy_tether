
function plotfig(name, titleg, xlabelg, ylabelg, xdata, ydata, legendg, showg)
    if nargin == 7
        showg = 0;
    end
    figure()
    title(titleg)
    xlabel(xlabelg)
    ylabel(ylabelg)
    plot(xdata, ydata, 'DisplayName', legendg)
    legend('show')
    saveas(gcf, name)
    if showg == 1
        show();
    else
        close();
    end
        
end