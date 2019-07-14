function [weight, mu, sigma] = em_gmm(weight, mu, sigma, data, N)
for i=1:N
  k = length(weight);
  expectation = nan(size(data,2),k);
  for i=1:size(data,2)
    p_log = sum(log(normpdf(data(:,i),mu,sigma)))+log(weight);
    p_log = p_log-max(p_log);
    p = exp(p_log);
    p_sum = sum(p);
    if p_sum > 0
      expectation(i,:) = p./sum(p);
    else
      expectation(i,:) = ones(size(p))./numel(p);
    end
  end
  for i=1:k
    weight(i) = sum(expectation(:,i))./size(data,2);
    if weight(i) > 0
      p = expectation(:,i)./sum(expectation(:,i));
      mu(:,i) = sum(data.' .* p);
      sigma(:,i) = sqrt(sum((data.'-mu(:,i).').^2 .* p));
    else 
      mu(:,i) = 0;
      sigma(:,i) = inf;
    end
    sigma = max(sigma,0.01);
  end
  valid = weight > 0.01;
  weight = weight(valid);
  mu = mu(:,valid);
  sigma = sigma(:,valid);
  weight = weight./sum(weight);
end