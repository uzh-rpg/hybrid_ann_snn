function test_dimred(X, kernel, kernel_param, Napp)

Kdet = EvalKernel(X,X,kernel, kernel_param);

% compute kpca
% low rank kernel approximation
[lambda U V] = lr_approx(Kdet,'eigdec');

mean_norm = zeros(1,min(size(lambda)));
max_norm = zeros(1,min(size(lambda)));
for i = 1 : min(size(lambda))
  mean_norm(i) = mean(mean(abs(Kdet - U(:,1:i)*lambda(1:i,1:i) * V(1:i,:))));
  max_norm(i) = max(max(abs(Kdet - U(:,1:i)*lambda(1:i,1:i) * V(1:i,:))));
end

plot(mean_norm); hold on; plot(max_norm,'color','k');
h = legend('Mean Norm (KPCA)','Max Norm (KPCA)'); set(gca, 'fontsize',16); set(h,'location','best','fontsize',16);
% compute pca


end