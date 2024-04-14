function plot_classification_perf()
x = [100         250         500         750        1000        1250        1500        1750        2000  2250        2500        2750        3000        3250        3500];
y = [ 0.1513    0.1807    0.2289    0.2490    0.2544    0.2677    0.2798    0.2784    0.2892    0.2932    0.2945  0.2972    0.3066    0.3119    0.3106];
z = .32;
figure;
plot(x,100*y, 'linewidth',4); hold on;
plot(x, 100*ones(length(x),1)* z,'--','linewidth',4);
set(gca,'linewidth',4, 'fontsize',20);
h = legend('\chi^2_{skewed}','RF approx. to \chi^2_{skewed}');
set(h,'fontsize',20,'location','best');
ylabel('Classification accuracy');xlabel('Number of RF features');
ylim([14 33]);xlim([0, 3500]);
end